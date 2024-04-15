from typing import Union
import os
import zarr
import s3fs
import dask.array as da
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
import re
import sunpy.visualization.colormaps

# setting configs
# matplotlib.use("Agg")


class SDOImageFetcher:
    @staticmethod
    def get_s3_connection(path_to_zarr: os.path) -> s3fs.S3Map:
        """
        Instantiate connection to aws for a given path `path_to_zarr`
        """
        return s3fs.S3Map(
            root=path_to_zarr,
            s3=s3fs.S3FileSystem(anon=True),
            # anonymous access requires no credentials
            check=False,
        )

    @staticmethod
    def load_single_aws_zarr(
        path_to_zarr: os.path,
        cache_max_single_size: int = None,
    ) -> Union[zarr.Array, zarr.Group]:
        """
        load zarr from s3 using LRU cache
        """
        return zarr.open(
            zarr.LRUStoreCache(
                store=SDOImageFetcher.get_s3_connection(path_to_zarr),
                max_size=cache_max_single_size,
            ),
            mode="r",
        )

    @staticmethod
    def get_single_solar_image(image_idx, path_to_zarr):
        images_drry = da.from_array(
            SDOImageFetcher.load_single_aws_zarr(path_to_zarr)["171A"]
        )
        image = np.array(images_drry[image_idx, :, :])
        return image

    @staticmethod
    def get_sdo_solar_images_from_aws(
        s3_root_for_sdoml_year_zarr,
        desired_times,
        sav_folder_path,
        tolerance,
        is_verbose=False,
    ):

        # for desired_times, get closest times in the zarr file and corresponding indices:
        #   images_zry_closest_idxs, images_closest_times
        images_zry_closest_idxs = []
        images_171a_zarray = SDOImageFetcher.load_single_aws_zarr(
            path_to_zarr=s3_root_for_sdoml_year_zarr,
        )["171A"]
        images_zry_times = pd.to_datetime(np.array(images_171a_zarray.attrs["T_OBS"]))

        # TEMP: pick up images_zry_times from local
        # pickle.dump(images_zry_times, open('temp_images_zry_times.pkl', 'wb'))
        # images_zry_times = pickle.load(open("temp_images_zry_times.pkl", "rb"))

        for desired_time in desired_times[None:None]:
            images_zry_closest_idx = np.argmin(abs(images_zry_times - desired_time))
            images_zry_closest_time = images_zry_times[images_zry_closest_idx]
            delta_time = abs(images_zry_closest_time - desired_time)
            if delta_time < tolerance:
                images_zry_closest_idxs.append(images_zry_closest_idx)
        images_zry_closest_idxs = sorted(set(images_zry_closest_idxs))
        images_closest_times = images_zry_times[images_zry_closest_idxs]

        # get the image_times that have been processed already: images_processed_times
        images_png_folder = sav_folder_path
        images_processed_paths = glob.glob(os.path.join(images_png_folder, "*.png"))
        images_processed_times = [
            pd.to_datetime(re.sub(".png", "", os.path.basename(path)))
            for path in images_processed_paths
        ]

        # fetch images
        fetched_images_paths = []
        for image_time in images_closest_times[None:None]:
            current_img_time = image_time

            # get the position of image_time in images_closest_times
            image_time_idx = list(images_closest_times).index(image_time)

            # check if the images_processed_times contains the row currently being processed and skip iter if true
            if current_img_time in images_processed_times:
                if is_verbose:
                    print(
                        f"Skipping image_time_num {image_time_idx + 1} as it has been processed already."
                    )
                continue

            # get current image
            image_arr = SDOImageFetcher.get_single_solar_image(
                images_zry_closest_idxs[image_time_idx], s3_root_for_sdoml_year_zarr
            )
            downsampled_pxl_posns = np.arange(0, image_arr.shape[0], 2)
            image_arr = image_arr[downsampled_pxl_posns, :][:, downsampled_pxl_posns]

            # Save the image
            image_path = f"{images_png_folder}/{current_img_time}.png"
            with plt.rc_context({'backend': 'Agg'}):
                plt.figure(figsize=(5, 5))
                plt.imshow(
                    image_arr,
                    origin="lower",
                    vmin=10,
                    vmax=1000,
                    cmap=plt.get_cmap("sdoaia171"),
                )
                plt.savefig(image_path)
                plt.close("all")
            
            if is_verbose:
                print(
                    f"fetched image_time_num: {image_time_idx + 1} of {len(images_closest_times)}"
                )

            fetched_images_paths.append(image_path)

        return fetched_images_paths