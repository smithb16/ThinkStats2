## Boilerplate and module import

import nsfg
import thinkstats2
import thinkplot

if __name__ == '__main__':
    """Main scripts"""
    ## Turn pregnancy data into DataFrame
    preg = nsfg.ReadFemPreg()
    resp = nsfg.ReadFemResp()
    live = preg[preg.outcome == 1]

    ## Test Mode method

    def Mode(self):
        """Gets the most common value in the distribution.

        return: float index
        """
        _, val = max((count, val) for val, count in self.Items())
        return val

    def AllMode(self):
        """Gets the value-frequency pairs in descending order.

        return: list of tuple of (value, count)
        """
        def second(tup):
            """Returns second element in tuple."""
            return tup[1]

        t = [(val, count) for val, count in self.Items()]
        return sorted(t, key=second, reverse=True)

    hist = thinkstats2.Hist(live.prglngth)
    #thinkplot.Hist(hist)
    #thinkplot.Show(xlabel='weeks',ylabel='count')
    print('Mode(hist):\n', Mode(hist))
    print('AllMode(hist):\n', AllMode(hist))

    ## Measure difference between firsts and others
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]

    first_hist = thinkstats2.Hist(np.floor(firsts.totalwgt_lb), label='first')
    other_hist = thinkstats2.Hist(np.floor(others.totalwgt_lb), label='other')

    print('firsts.totalwgt_lb.mean():\n', firsts.totalwgt_lb.mean())
    print('others.totalwgt_lb.mean():\n', others.totalwgt_lb.mean())

    d = thinkstats2.CohenEffectSize(firsts.totalwgt_lb, others.totalwgt_lb)
    print('effect size:\n', d)
    width = 0.45
    thinkplot.PrePlot(2)
    thinkplot.Hist(first_hist, align='right', width=width)
    thinkplot.Hist(other_hist, align='left', width=width)
    thinkplot.Show(xlabel='lb', ylabel='frequency', xlim=[-1,15])
