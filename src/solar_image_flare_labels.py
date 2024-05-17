import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

class SDOImageFlareTagging:
    @staticmethod
    def get_is_big_flare(solar_image_path, big_flare_max_dttimes, tolerance):

        is_big_flare = False

        # get datetime of solar_image_path: solar_image_datetime
        solar_image_fname = os.path.basename(solar_image_path)
        solar_image_datetime = pd.to_datetime(
            solar_image_fname[0 : solar_image_fname.rfind(".png")]
        ).replace(tzinfo=None)

        # compare solar_image_datetime with max_datetimes and set is_big_flare to True
        # if solar_image_datetime within some tolerance of max_datetimes
        min_time_delta = min(abs(solar_image_datetime - big_flare_max_dttimes))
        if min_time_delta < tolerance:
            is_big_flare = True

        return is_big_flare    
    
    @staticmethod
    def get_big_flare_labels(solar_images_folder, goes_file_path):
        # init result
        big_flare_labels_df = pd.DataFrame(
            columns=["solar_image_filename", "is_big_flare"]
        )

        # set tolerances for sure_big_flare and unsure_big_flare
        sure_bigflare_tol = pd.to_timedelta(5, "min")
        unsure_bigflare_tol = pd.to_timedelta(24, "h")

        # get big_flare_max_dttimes
        goes_events_df = pd.read_csv(goes_file_path)
        goes_mx_events = goes_events_df[
            goes_events_df["Particulars_a"].str.lower().str.startswith(("m", "x"))
        ]
        big_flare_max_dttimes = pd.to_datetime(goes_mx_events["max_datetime"]).dropna()

        # get solar_image_paths
        solar_image_paths = glob.glob(f"{solar_images_folder}/*png")

        # for each solar image path, get is_big_flare and populate df
        time_tolerance_minutes = sure_bigflare_tol.total_seconds() / 60
        for i, solar_image_path in enumerate(solar_image_paths[0:None]):
            # assume is_big_flare is 0 (not a big flare)
            is_big_flare = 0

            # setting the is_big_flare value
            # - if within sure_bigflare_tol, we set it to 1
            # - if not within sure_bigflare_tol and within unsure_bigflare_tol, we set to None
            if SDOImageFlareTagging.get_is_big_flare(solar_image_path, big_flare_max_dttimes, sure_bigflare_tol):
                is_big_flare = 1
            elif SDOImageFlareTagging.get_is_big_flare(
                solar_image_path, big_flare_max_dttimes, unsure_bigflare_tol
            ):
                is_big_flare = None

            big_flare_labels_df.loc[len(big_flare_labels_df)] = (
                os.path.basename(solar_image_path),
                is_big_flare,
            )

        return big_flare_labels_df
    
    @staticmethod
    def plot_solar_images(solar_images_folder,plot_df, axs):
        for i, ax in enumerate(axs.flat):
            solar_image_path = (
                f"{solar_images_folder}/{plot_df.iloc[i]['solar_image_filename']}"
            )
            solar_image = Image.open(solar_image_path)
            ax.imshow(solar_image)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    