import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import numpy as np
from matplotlib.widgets import Cursor

def plot_tracks_enhanced(tracks, track_id_list):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Track Measurements and State Estimates')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use a color map for better distinction

    all_times = []
    all_data = []

    for track_id, track in enumerate(tracks):
        if track_id_list[track_id]['state'] == 'occupied':
            color = colors[track_id % len(colors)]
            
            times = [datetime.datetime.fromtimestamp(m[0][3]) for m in track['measurements']]
            ranges = [m[0][0] for m in track['measurements']]
            azimuths = [m[0][1] for m in track['measurements']]
            elevations = [m[0][2] for m in track['measurements']]
            
            sf_ranges = [m[1][0] if len(m) > 1 else None for m in track['measurements']]
            sf_azimuths = [m[1][1] if len(m) > 1 else None for m in track['measurements']]
            sf_elevations = [m[1][2] if len(m) > 1 else None for m in track['measurements']]

            # Plot range
            ax1.scatter(times, ranges, c=[color], marker='o', label=f'Track {track_id} (Measured)')
            ax1.plot(times, sf_ranges, c=color, linestyle='--', label=f'Track {track_id} (Estimated)')

            # Plot azimuth
            ax2.scatter(times, azimuths, c=[color], marker='o')
            ax2.plot(times, sf_azimuths, c=color, linestyle='--')

            # Plot elevation
            ax3.scatter(times, elevations, c=[color], marker='o')
            ax3.plot(times, sf_elevations, c=color, linestyle='--')

            # Store data for cursor
            all_times.extend(times)
            all_data.extend([(track_id, r, a, e, t) for r, a, e, t in zip(ranges, azimuths, elevations, times)])

    # Set labels and formats
    ax1.set_ylabel('Range')
    ax2.set_ylabel('Azimuth')
    ax3.set_ylabel('Elevation')
    ax3.set_xlabel('Time')

    # Format time axis
    date_formatter = DateFormatter("%H:%M:%S")
    ax1.xaxis.set_major_formatter(date_formatter)
    ax2.xaxis.set_major_formatter(date_formatter)
    ax3.xaxis.set_major_formatter(date_formatter)

    # Add legends
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Use tight layout
    plt.tight_layout()

    # Create cursor for data tips
    cursor1 = Cursor(ax1, useblit=True, color='red', linestyle='--')
    cursor2 = Cursor(ax2, useblit=True, color='red', linestyle='--')
    cursor3 = Cursor(ax3, useblit=True, color='red', linestyle='--')

    # Convert all times to numbers for easier comparison
    all_times_num = [t.timestamp() for t in all_times]

    def on_click(event):
        if event.inaxes:
            # Find the closest point
            diff = np.abs(np.array(all_times_num) - event.xdata)
            idx = diff.argmin()
            
            if diff[idx] <= 0.1:  # Within 0.1 seconds
                track_id, r, a, e, t = all_data[idx]
                text = f'Track ID: {track_id}\nRange: {r:.2f}\nAzimuth: {a:.2f}\nElevation: {e:.2f}\nTime: {t}'
                event.inaxes.annotate(text, (event.xdata, event.ydata), xytext=(10, 10), 
                                      textcoords='offset points', bbox=dict(boxstyle='round', fc='w'),
                                      arrowprops=dict(arrowstyle='->'))
                plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Save the plot
    plt.savefig('enhanced_track_plots.png')
    print("Enhanced track plots have been saved as 'enhanced_track_plots.png'")
    
    plt.show()

# Usage example:
# plot_tracks_enhanced(tracks, track_id_list)