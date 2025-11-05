import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({'font.family': 'Arial'})

fig = plt.figure(figsize=(12, 6), dpi=200)
axes = fig.subplots(1, 2)
b_w = 1.3

step = np.array([
    1, 25001, 50001, 75001, 100001, 125001, 150001, 175001, 200001, 225001, 250001, 275001,
    300001, 325001, 350001, 375001, 400001, 425001, 450001, 475001, 500000
])

ax = axes[0]

hc_run1 = np.array([
    2.2089469898075387, 22.060823070095818, 43.46746865566417, 37.5516712159563, 42.59676727334465,
    46.08326961737371, 45.316709911793865, 42.61256363715845, 44.26733314032958, 45.4083807530177,
    45.92270877719229, 42.39063168212256, 44.68315267818741, 43.7841020120801, 45.21967660672732,
    40.06958960721873, 40.977270834044006, 46.79343941745763, 46.51313559939821, 46.47920232835251,
    39.190706132956294
])
hc_run2 = np.array([
    2.241110196128494, 8.415312647561635, 23.488336360894234, 36.21612603812008, 20.09057436244617,
    20.16645009663261, 41.96699027897095, 46.00738574537723, 36.04469166690424, 39.8155187246612,
    44.92550585654143, 45.47853685482725, 40.99886979640412, 46.44381654613025, 47.513719801570346,
    36.31935851420119, 40.32189885766686, 48.31565307086411, 46.33754155757789, 45.69184578529377,
    46.474857718787575
])

privorl_runs = [hc_run1, hc_run2]
privorl_stack = np.vstack(privorl_runs)
privorl_mean = privorl_stack.mean(axis=0)
privorl_lower = privorl_stack.min(axis=0)
privorl_upper = privorl_stack.max(axis=0)

real_run1 = np.array([
    2.227126460409432, 36.14962713222954, 32.43995983105152, 35.46671008768045, 46.5162727764802,
    47.47500337199374, 46.89520392210521, 48.08100755421931, 49.0613984932881, 47.53993434629726,
    46.42931919823131, 47.482891657991115, 44.51684177871727, 47.05665285403946, 47.99804435606274,
    48.05849684063392, 46.93417547534224, 48.58338591877585, 47.89729952341549, 48.68535277381897,
    48.69267533151155
])
real_run2 = np.array([
    2.24508086379671, 38.99986572847394, 43.29012210557389, 46.31002195517985, 47.03715022125242,
    47.05835257819792, 47.23747025188057, 47.45030435286393, 47.98309677284367, 43.29341786228645,
    48.29392595919668, 47.29185244605582, 48.97015077666035, 47.1062542734244, 47.6583657473972,
    48.07432998615475, 48.15432200096642, 44.71987802309216, 42.88242050232679, 48.74452969755449,
    48.36598202938857
])
real_run3 = np.array([
    2.2093192784527056, 41.68794825815374, 39.00840909728514, 41.7236578040272, 47.79380072763817,
    47.77118070737092, 47.03661232160539, 47.54641102026911, 47.13698351155485, 46.99981178771761,
    47.36059575002318, 48.62160988648602, 47.34394455717856, 48.42996940217686, 48.4097862034896,
    47.61314083547284, 48.97039306269094, 48.54781119151879, 48.74036851660103, 48.263206343823,
    49.24373119009369
])
real_runs = [real_run1, real_run2, real_run3]
real_stack = np.vstack(real_runs)
real_mean = real_stack.mean(axis=0)
real_lower = real_stack.min(axis=0)
real_upper = real_stack.max(axis=0)

ax.grid(color='lightgrey', linewidth=1, zorder=0)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

ax.spines['bottom'].set_linewidth(b_w)
ax.spines['left'].set_linewidth(b_w)
ax.spines['top'].set_linewidth(b_w)
ax.spines['right'].set_linewidth(b_w)

ax.plot(step, privorl_mean, label='PrivORL-n', linewidth=3.5, color='#82B0D2',
        marker='X', markersize=10, markerfacecolor='#82B0D2', markeredgecolor='white', zorder=100)
ax.fill_between(step, privorl_lower, privorl_upper, alpha=0.25, color='#82B0D2', zorder=50)

ax.plot(step, real_mean, label='Real', linewidth=3.5, color='#FA7F6F',
        marker='s', markersize=8, markerfacecolor='#FA7F6F', markeredgecolor='white', zorder=100)
ax.fill_between(step, real_lower, real_upper, alpha=0.25, color='#FA7F6F', zorder=50)

ax.set_xlabel('Step', size=24, weight='bold')
ax.set_ylabel('Normalized Return', size=24, weight='bold')
ax.set_xticks([0, 1e5, 2e5, 3e5, 4e5, 5e5])
ax.set_xticklabels(['0', '1e5', '2e5', '3e5', '4e5', '5e5'])
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(18)
    label.set_fontweight('bold')
ax.legend(loc='lower right', prop={'size': 20, 'weight': 'bold'})
ax.text(0.5, -0.22, 'Mujoco-halfcheetah', ha='center', va='center', fontsize=28, weight='bold', transform=ax.transAxes)
ax.set_ylim(bottom=0, top=55)

ax = axes[1]

def smooth(y, window=1):
    return np.convolve(y, np.ones(window)/window, mode='same')

medium_run1 = np.array([
    -10.504427662640355, 25.98040708133219, 44.568477854954544, 32.13033771770799, 20.00947595100472,
    7.745671968363036, 47.84692844753016, 48.47600862800078, 32.67250378808589, 29.2577774346039,
    48.08317539659762, 42.6234125741829, 53.91841874354055, 48.58001619568743, 37.475426440086192,
    45.37785874504964, 44.70488653724762, 48.179140021567925, 44.435895900422686, 57.48239627263729,
    54.75349438537112
])
medium_run2 = np.array([
    9.367082763393556, 32.0499067411106, 48.15280465377679, 32.3368069047225, 37.90870895690313,
    33.12712489375381, 47.21380264101294, 41.459450456134505, 47.47308773473381, 54.5194235756191,
    49.1471786464028, 60.97689998222027, 43.7712141748585, 53.96608635758913, 45.609343129016864,
    53.79610219068518, 47.48200536914397, 50.88709981268318, 48.33690195899304, 45.24034331181016,
    50.38876108732877
])
medium_run3 = np.array([
    -3.426742355782697, 40.44002910884477, 39.92902579127276, 32.82946226249146, 34.67083859719225,
    39.64043341997253, 29.429360246055143, 48.01389975690779, 42.44228627443768, 33.00464117983722,
    50.0250121206129, 40.380801755179675, 37.35186720230786, 32.07121165009511, 29.577345085834523,
    48.25362455431582, 44.566822209742185, 50.11798123391047, 48.244307901361694, 44.19213313627166,
    40.35593721743444
])

medium_run1 = smooth(medium_run1)
medium_run2 = smooth(medium_run2)
medium_run3 = smooth(medium_run3)

privorl_runs_medium = [medium_run1, medium_run2, medium_run3]
privorl_stack_medium = np.vstack(privorl_runs_medium)
privorl_mean_medium = privorl_stack_medium.mean(axis=0)
privorl_lower_medium = privorl_stack_medium.min(axis=0)
privorl_upper_medium = privorl_stack_medium.max(axis=0)

real_run1_medium = np.array([
    -10.504427662640357, 35.72363594973147, 37.88606122012192, 49.9106052798443, 46.76992426991134,
    35.040769635164715, 46.12513081026475, 46.41140597480655, 48.949556895587655, 46.9299097816334,
    49.28496664613679, 48.57043139997384, 47.05621182376139, 42.395087513929425, 49.89032014430069,
    44.21660666215192, 49.67997396745739, 46.27842875213452, 50.23380316997746, 46.27228035595207,
    58.50400795195916
])
real_run2_medium = np.array([
    -3.426742355782697, 43.30248283499248, 37.87137541592992, 42.919689320171905, 12.849343131615102,
    51.60562709362437, 40.12925900186943, 24.941182112983533, 38.66187002151955, 38.372801863455244,
    41.85328991917401, 61.37095924602318, 46.69089676594321, 46.54532069426382, 59.142835820798666,
    44.57883415257352, 44.073720215096465, 45.2013515216786, 55.89697072087708, 51.21704675480657,
    47.25027434773612
])
real_run3_medium = np.array([
    5.457089400267115, 33.58832815494398, 42.77856857293771, 50.774452251656655, 54.00812150808445,
    21.078207234036444, 44.527691803950084, 41.75819367472467, 49.93526840282051, 45.08956422881987,
    45.72360768367486, 46.91060478934008, 46.875875479462984, 50.70952746615097, 48.607177086801364,
    45.37671165229646, 47.47448048778449, 50.163905565321485, 50.27347492596817, 53.81746608773239,
    50.38465190489562
])

real_run1_medium = smooth(real_run1_medium)
real_run2_medium = smooth(real_run2_medium)
real_run3_medium = smooth(real_run3_medium)

real_runs_medium = [real_run1_medium, real_run2_medium, real_run3_medium]
real_stack_medium = np.vstack(real_runs_medium)
real_mean_medium = real_stack_medium.mean(axis=0)
real_lower_medium = real_stack_medium.min(axis=0)
real_upper_medium = real_stack_medium.max(axis=0)

ax.grid(color='lightgrey', linewidth=1, zorder=0)
ax.spines['bottom'].set_linewidth(b_w)
ax.spines['left'].set_linewidth(b_w)
ax.spines['top'].set_linewidth(b_w)
ax.spines['right'].set_linewidth(b_w)

ax.plot(step, privorl_mean_medium, label='PrivORL-n', linewidth=3.5, color='#82B0D2',
        marker='X', markersize=10, markerfacecolor='#82B0D2', markeredgecolor='white', zorder=100)
ax.fill_between(step, privorl_lower_medium, privorl_upper_medium, alpha=0.25, color='#82B0D2', zorder=50)

ax.plot(step, real_mean_medium, label='Real', linewidth=3.5, color='#FA7F6F',
        marker='s', markersize=8, markerfacecolor='#FA7F6F', markeredgecolor='white', zorder=100)
ax.fill_between(step, real_lower_medium, real_upper_medium, alpha=0.25, color='#FA7F6F', zorder=50)

ax.set_xlabel('Step', size=24, weight='bold')
ax.set_xticks([0, 1e5, 2e5, 3e5, 4e5, 5e5])
ax.set_xticklabels(['0', '1e5', '2e5', '3e5', '4e5', '5e5'])
ax.set_ylim(bottom=-5, top=70)
ax.set_yticks([-5, 10, 25, 40, 55, 70])
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(18)
    label.set_fontweight('bold')
ax.legend(loc='lower right', prop={'size': 20, 'weight': 'bold'})
ax.text(0.5, -0.22, 'Maze2d-medium', ha='center', va='center', fontsize=28, weight='bold', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('fig_real_and_-n_curve/agent_traj.pdf', dpi=200)
plt.savefig('fig_real_and_-n_curve/agent_traj.png', dpi=200)
