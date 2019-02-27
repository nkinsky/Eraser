# Behavioral plots here!

import eraser_reference as err
import er_plot_functions as er
import scipy.stats as s

## Get difference in context-specific freezing between groups at LTM memory time points!
_, _, fratio_cont= er.plot_all_freezing(err.control_mice_good)
_, _, fratio_ani= er.plot_all_freezing(err.ani_mice_good)
stats, pval = s.ttest_ind(fratio_ani[1,3:6,:].reshape(-1) - fratio_ani[0,3:6,:].reshape(-1),
                          fratio_cont[1,3:6,:].reshape(-1) - fratio_cont[0,3:6,:].reshape(-1), nan_policy='omit')