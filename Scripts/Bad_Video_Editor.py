#get user input for session number
from os import path
mouse = input("What mouse are you correcting?")
type(mouse)
day = input("What day corresponds to the session you are correcting?")
type(day)
sesh = input('What session does this correspond to?')
type(sesh)
#fetch the bad points
import er_plot_functions as er
bad_frames = er.get_bad_epochs(str(mouse),'Open',day)
print(bad_frames)
#verify that the session in question is in the directory
import session_directory as sd
sd.check_session(int(sesh))
#load previously coded video
import ff_video_fixer as f
import matplotlib.pyplot as plt
ff = f.load_movie(int(sesh))
# new_directory = r'E:\Evan\Imaging\Marble_11\20180528_1_openfield\freezeframe'
# ff.location = path.join(new_directory, 'Movie.pkl')
# ff.avi_location = path.join(new_directory, '20180528_1_openfield_Marble_11_video_crop.AVI')
# ff.csv_location = path.join(new_directory, 'pos.csv')
ff.correct_position()


ff.save_data()
ff.export_pos()
print("Fixing successfully finished")