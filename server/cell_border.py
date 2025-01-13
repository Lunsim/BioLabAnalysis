@app.post("/spg", response_model=xxx)
def read_and_store_images(sub_folder, image_list):
    folder_path = os.path.join(base_directory, sub_folder)
    filenames = os.listdir(folder_path)
    sorted_filenames = sorted(filenames, key=lambda x: int(re.search(r'(\d+)', x).group()))

    for filename in sorted_filenames:
        if filename.endswith('.czi'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            img = AICSImage(file_path)
            image_data = img.get_image_data("YX")
            #uint8_stacked = im_adjust(image_data, autoscale = 'uint8')
            image_list.append(image_data)