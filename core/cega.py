import matplotlib.pyplot as plt


def determine_zone(actual, pred):
    """
    Determines CEGA zone for given scatter point
    :param actual: Reference value
    :param pred: Predicted value
    :return: CEGA zone where the given scatter point should lay
    """
    #Zone A
    if (actual <= 70 and pred <= 70) or 1.2 * actual >= pred >= 0.8 * actual:
        return 0

    # Zone E - left upper
    if actual <= 70 and pred >= 180:
       return 4
    # Zone E - right lower
    if actual >= 180 and pred <= 70:
        return 4

    # Zone C - upper
    if 70 <= actual <= 290 and pred >= actual + 110:
        return 2
    # Zone C - lower
    if 130 <= actual <= 180 and pred <= (7/5) * actual - 182:
        return 2

    # Zone D - right
    if actual >= 240 and 70 <= pred <= 180:
        return 3
    # Zone D - left
    if actual <= 70 <= pred <= 180:
        return 3

    # Zone B
    else:
        return 1


def clarke_error_grid(ref_values, pred_values, title_string, save_file=None, plot=True):
    """
    Constructs a Clarke error grid plot and calculate zone statistics
    :param ref_values: Array of reference BG values
    :param pred_values: Array of predicted BG values
    :param title_string: Title for the plot
    :return: CEGA scatter plot and percentage of points within CEGA zones
    """
    MAX = 410

    assert len(ref_values) == len(pred_values), f"Got {len(ref_values)} reference values nad {len(pred_values)} predicted values, values must match."

    assert max(ref_values) <= MAX and max(pred_values) <= MAX, "Predicted or reference BG values are above valid range."
    assert min(ref_values) > 0 and min(pred_values) > 0, "Predicted or reference BG values are below valid range."

    # Clear
    fig = None
    if plot:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

        ax.scatter(ref_values, pred_values, marker='o', color='orange', s=8)
        ax.set_title(f"{title_string} CEGA", fontsize=17)
        ax.set_xlabel("Reference Concentration [mg/dl]", fontsize=11)
        ax.set_ylabel("Predicted Concentration [mg/dl]", fontsize=11)
        ax.set_xticks([tick for tick in range(0, MAX, 50)])
        ax.set_yticks([tick for tick in range(0, MAX, 50)])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_facecolor('white')

        # Set axes lengths
        ax.set_xlim([0, MAX])
        ax.set_ylim([0, MAX])
        ax.set_aspect(MAX / MAX)

        # Zone lines
        # Regression line
        ax.plot([0, 400], [0, 400], ':', c='black')

        # Zone A
        ax.plot([0, 175/3], [70, 70], '-', c='black')
        ax.plot([175/3, 400/1.2], [70, 400], '-', c='black')
        ax.plot([70, 70], [0, 56], '-', c='black')
        ax.plot([70, 400], [56, 320], '-', c='black')

        # Zone B
        ax.plot([70, 70], [0, 56], '-', c='black')

        # Zone D and E
        ax.plot([70, 70], [84, 400], '-', c='black')
        ax.plot([0, 70], [180, 180], '-', c='black')
        # E
        ax.plot([180, 400], [70, 70], '-', c='black')
        # D
        ax.plot([240, 240], [70, 180], '-', c='black')
        ax.plot([240, 400], [180, 180], '-', c='black')

        # Zone C
        ax.plot([70, 290], [180, 400], '-', c='black')
        ax.plot([180, 180], [0, 70], '-', c='black')
        ax.plot([130, 180], [0, 70], '-', c='black')


        # Add zone titles
        zone_title_positions = {
            "A": [(30, 15), (360, 320), (320, 360)],
            "B": [(370, 260), (280, 370)],
            "C": [(160, 370), (160, 15)],
            "D": [(30, 140), (370, 120)],
            "E": [(30, 370), (370, 15)]
        }
        for zone, positions in zone_title_positions.items():
            for pos in positions:
                ax.text(pos[0], pos[1], zone, fontsize=15)

    # Calculate statistics
    zones = [0] * 5
    for i in range(len(ref_values)):
        zones[determine_zone(ref_values[i], pred_values[i])] += 1
    total = sum(zones)
    # Get percentage
    zones = [zone / total for zone in zones]

    if save_file and plot:
        fig.tight_layout(pad=0.5)
        fig.savefig(fname=f"{save_file}")
    return fig, zones
