# region INPUT DATA

# original = {'chamfer': {'epi': (0.7819856453797468,
#                                 0.14406961574306296),
#                         'lv': (0.6685212932278481,
#                                 0.19142234569387034),
#                         'rv': (0.7084630520886077,
#                                 0.16660997680666448)},
#             'haussdorff': {'epi': (4.383336975100178,
#                                     1.4061162686906767),
#                             'lv': (3.4752692133626346,
#                                 1.4937976467937415),
#                             'rv': (3.724932416992789,
#                                 1.6429252264674568)}}

original = {'chamfer': {'epi': (0.71304,
                                0.12139),
                        'lv': (0.62123,
                                0.18620),
                        'rv': (0.63375,
                                0.13323)},
            'haussdorff': {'epi': (3.86309,
                                    1.20920),
                            'lv': (3.17575,
                                1.45383),
                            'rv': (3.29959,
                                1.40665)}}

case3_3srf_from2_and_sdf_filter = {
    'SDF threshold: 002': {'epi+lv': {'chamfer': {'epi': (0.7322733523417722,
                                                0.12276004041586361),
                                        'lv': (0.5894172869620253,
                                               0.176293230257814),
                                        'rv': (0.9860221775316453,
                                               0.29467787126199096)},
                            'haussdorff': {'epi': (4.047530171160143,
                                                   1.2769696061531715),
                                           'lv': (2.9790727495548546,
                                                  1.3797481653516581),
                                           'rv': (5.477512252138449,
                                                  4.326725156880454)}},
                 'epi+rv': {'chamfer': {'epi': (0.7161246936708862,
                                                0.12736965918974572),
                                        'lv': (1.5897370155696204,
                                               0.7362251850814177),
                                        'rv': (0.6013517690506329,
                                               0.13761366717906395)},
                            'haussdorff': {'epi': (3.9607037150533344,
                                                   1.4580723690171509),
                                           'lv': (5.884543888741281,
                                                  3.2759820832189805),
                                           'rv': (3.0865071204424073,
                                                  1.5372596843589843)}},
                 'lv+rv': {'chamfer': {'epi': (1.0941498641139242,
                                               0.29836363157438395),
                                       'lv': (0.6156997126582279,
                                              0.19527692462270105),
                                       'rv': (0.646919792721519,
                                              0.17483708918403862)},
                           'haussdorff': {'epi': (5.269239456332061,
                                                  1.7802821090384686),
                                          'lv': (3.169405241846965,
                                                 1.3286039927514466),
                                          'rv': (3.357860323545287,
                                                 1.5633637484091432)}}},

    'SDF threshold: 001': {'epi+lv': {'chamfer': {'epi': (0.7592644241772152,
                                                0.12708351406078472),
                                           'lv': (0.5967727224683544,
                                               0.18013268333582474),
                                           'rv': (1.019630085443038,
                                               0.29546891566057193)},
                               'haussdorff': {'epi': (4.444816888422233,
                                                   1.4708160241766373),
                                           'lv': (3.0260393923403686,
                                                  1.3975709636802038),
                                           'rv': (5.76080904164458,
                                                  4.217126356584917)}},
                    'epi+rv': {'chamfer': {'epi': (0.737090624113924,
                                                0.13318423134590757),
                                           'lv': (1.6114643878481012,
                                               0.7364115152702659),
                                           'rv': (0.6139894620253165,
                                               0.1397216488102912)},
                               'haussdorff': {'epi': (4.152831026730968,
                                                   1.4768394337032238),
                                              'lv': (5.9232856323015355,
                                                  3.220377018933032),
                                              'rv': (3.1535334676717923,
                                                  1.5758191290712893)}},
                    'lv+rv': {'chamfer': {'epi': (1.1143767160759495,
                                               0.3090155374988607),
                                          'lv': (0.6312143602531646,
                                              0.19895870779289948),
                                          'rv': (0.6660696739873417,
                                              0.18220003795672865)},
                              'haussdorff': {'epi': (5.352588004589717,
                                                  1.8546048573114942),
                                             'lv': (3.299018269352276,
                                                 1.3662813086235828),
                                             'rv': (3.5280762776623864,
                                                 1.6418620462541327)}}},

'SDF threshold: 0005': {'epi+lv': {'chamfer': {'epi': (0.8274245109493671,
                                                 0.13849995469697218),
                                         'lv': (0.6158343465189874,
                                                0.18372714102068907),
                                         'rv': (1.107849213101266,
                                                0.3016471006817495)},
                             'haussdorff': {'epi': (5.036805015168894,
                                                    1.5802052825949457),
                                            'lv': (3.145583790989706,
                                                   1.4476908979737784),
                                            'rv': (6.11197995379866,
                                                   4.200534603210717)}},
                  'epi+rv': {'chamfer': {'epi': (0.7835425540506329,
                                                 0.14154462978531174),
                                         'lv': (1.6446078679113925,
                                                0.7237564854344967),
                                         'rv': (0.64683537,
                                                0.1491840233365904)},
                             'haussdorff': {'epi': (4.482811024992048,
                                                    1.5095611716695263),
                                            'lv': (6.019774268056313,
                                                   3.1873610003885724),
                                            'rv': (3.389218775415632,
                                                   1.628396829621716)}},
                  'lv+rv': {'chamfer': {'epi': (1.1523515394303796,
                                                0.3104124650723546),
                                        'lv': (0.6621880002531645,
                                               0.19757355190201054),
                                        'rv': (0.7168452546835442,
                                               0.19543587182236696)},
                            'haussdorff': {'epi': (5.620187219753993,
                                                   1.9278303026184764),
                                           'lv': (3.5557660679233236,
                                                  1.3673146474658506),
                                           'rv': (3.9748551372069034,
                                                  1.8424306394290495)}}}}
# endregion


import matplotlib.pyplot as plt
import numpy as np


def plot_combined_case(metric_name, original, combined_dict):

    thresholds = list(combined_dict.keys())

    surfaces = ['epi', 'lv', 'rv']
    x = np.arange(len(surfaces))

    fig, axes = plt.subplots(
        1,
        len(thresholds),
        figsize=(6 * len(thresholds), 6),
        sharey=True
    )

    if len(thresholds) == 1:
        axes = [axes]

    width = 0.18

    for ax, thresh_name in zip(axes, thresholds):

        comparison_dict = combined_dict[thresh_name]

        labels = ['original'] + list(comparison_dict.keys())

        for i, label in enumerate(labels):

            if label == 'original':
                data = original[metric_name]
            else:
                data = comparison_dict[label][metric_name]

            means = [data[s][0] for s in surfaces]
            stds = [data[s][1] for s in surfaces]

            offset = (i - (len(labels) - 1) / 2) * width

            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt='o',
                capsize=5,
                label=label
            )

        ax.set_xticks(x)
        ax.set_xticklabels(['epi', 'lv_endo', 'rv_endo'])

        ax.set_title(thresh_name)

        ax.grid(True)

    axes[0].set_ylabel(f"{metric_name} (mm)")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='upper right',
        ncol=4
    )

    fig.suptitle(
        f'{metric_name} - Combined Surface Inference + SDF Threshold',
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.show()


plot_combined_case(
    metric_name='chamfer',
    original=original,
    combined_dict=case3_3srf_from2_and_sdf_filter
)

plot_combined_case(
    metric_name='haussdorff',
    original=original,
    combined_dict=case3_3srf_from2_and_sdf_filter
)