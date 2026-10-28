"""Check that presentation contrasts retain losses and separate alignment."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from payoff_presentation_followup import compare_cells


class PresentationTests(unittest.TestCase):
    def test_crossed_contrasts_do_not_pool_away_a_reversal(self):
        rows=[]
        for condition in ('base','adapter','offset'):
            for column in (0,1):
                for name in (0,1):
                    for query in ('private','message','fact0','fact1'):
                        for alignment in ('aligned','opposed'):
                            correct = not (query=='private' and ((condition=='offset' and column==0) or (condition=='adapter' and column==1)))
                            recipient = alignment=='aligned' or (query=='message' and condition=='adapter')
                            rows.append({'condition':condition,'column_swap':column,'name_swap':name,'query':query,
                                         'alignment':alignment,'correct':correct,'recipient_optimal':recipient,
                                         'other_client_optimal':not correct,'dominated':False})
        cells, contrasts=compare_cells(rows)
        self.assertEqual(len(cells),96)
        self.assertEqual(sum(r['n'] for r in cells),len(rows))
        for r in contrasts:
            self.assertEqual(r['adapter_minus_offset_private'],1 if r['column_swap']==0 else -1)
            self.assertEqual(r['adapter_minus_base_opposed_recipient'],1)


if __name__=='__main__':unittest.main()
