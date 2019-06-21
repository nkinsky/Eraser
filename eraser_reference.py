# Marble14 should also be included once 10Hz SR is accounted for and Shock day 2 is run
control_mice = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble14', 'Marble24',
                'Marble26', 'Marble27', 'Marble29']

control_mice_good = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble14', 'Marble24',
                     'Marble27', 'Marble29']

ani_mice = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble21', 'Marble25']

ani_mice_good = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble25']  # 'Marble18 ?'

all_mice = control_mice.copy()
all_mice.append(ani_mice)

all_mice_good = control_mice_good.copy()
all_mice_good.append(ani_mice_good)

generalizers = ["Marble29", "Marble11", "Marble06"]

discriminators = ["Marble12", "Marble24", "Marble27", "Marble07"]


# Plotting folder
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'