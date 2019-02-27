# Functions to look at generalization of freezing between arenas
import er_plot_functions as er

def get_DIfreeze(mouse):
    """
    Describe code here
    :param mouse:
    :return:
    """
    fratios = er.get_all_freezing(mouse, day_des=[-2, -1, 4, 1, 2, 7], arenas=['Open', 'Shock'])


    # return DIfrz:

if __name__ == '__main__':
    get_DIfreeze('Marble29')
    pass