import { writeFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// The minimal harness's attempt is one run of the shown tests, so here a submission is one bash call running test.py.
// PI_MAX_SUBMISSIONS (0 = no cap), PI_TEST_TIMEOUT (seconds per test run, the minimal harness's VERIFY_TIMEOUT),
// PI_SAMPLING (JSON merged into every request), PI_BUDGET_LOG (summary file).
// Known issue (unfixed): a piped test run (`python test.py | head`) exits with the last command's status, so a failing
// run counts as "passed" and ends the episode (Qwen3.8 LCB Pro screen 2026-10-30: 2 false ends). Candidate fix: prefix
// test-run commands with `set -o pipefail;` in the tool_call handler.
const TEST_RUN = /\bpython[\d.]*\s+(?:-\S+\s+)*(?:\.\/)?test\.py\b/;

type EndReason = "passed" | "attempt_limit";

export default function (pi: ExtensionAPI) {
	const maxSubmissions = Number(process.env.PI_MAX_SUBMISSIONS ?? "0");
	const testTimeout = Number(process.env.PI_TEST_TIMEOUT ?? "0");
	const sampling: Record<string, unknown> = JSON.parse(process.env.PI_SAMPLING ?? "{}");
	const logPath = process.env.PI_BUDGET_LOG;
	const testRunCallIds = new Set<string>();
	let endReason: EndReason | null = null;

	const writeLog = () => {
		if (logPath) writeFileSync(logPath, JSON.stringify({ test_runs: testRunCallIds.size, end_reason: endReason }));
	};

	pi.on("before_provider_request", (event) => ({ ...(event.payload as object), ...sampling }));

	pi.on("tool_call", (event) => {
		if (endReason !== null) return { block: true, reason: "The episode is over." };
		const command = event.toolName === "bash" ? String((event.input as { command?: string }).command ?? "") : "";
		if (!TEST_RUN.test(command)) return;
		if (maxSubmissions > 0 && testRunCallIds.size >= maxSubmissions) {
			return { block: true, reason: "No submissions remaining." };
		}
		testRunCallIds.add(event.toolCallId);
		if (testTimeout > 0) (event.input as { timeout?: number }).timeout = testTimeout;
		writeLog();
	});

	pi.on("tool_result", (event, ctx) => {
		if (!testRunCallIds.has(event.toolCallId)) return;
		if (!event.isError) endReason = "passed";
		else if (maxSubmissions > 0 && testRunCallIds.size >= maxSubmissions) endReason = "attempt_limit";
		writeLog();
		if (endReason !== null) {
			ctx.abort();
			return;
		}
		if (maxSubmissions === 0) return;
		const remaining = `${maxSubmissions - testRunCallIds.size} of ${maxSubmissions} submissions remaining.`;
		return { content: [...event.content, { type: "text", text: `\n\n${remaining}` }] };
	});
}
