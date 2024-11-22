from fastapi import FastAPI

app = FastAPI()

##Section 1:
##
##
##
@app.post("/muscle", response_model=xxx)
def im_adjust(I, thres=[1, 99, True], autoscale=None):

    # compute percentile: remove too big or too small values
    # thres: [thres_low, thres_high, percentile]
    if thres[2]:
        I_low, I_high = np.percentile(I.reshape(-1), thres[:2])
    else:
        I_low, I_high = thres[0], thres[1]
    # thresholding
    I[I > I_high] = I_high
    I[I < I_low] = I_low
    if autoscale is not None:
        # scale to 0-1
        I = (I.astype(float) - I_low) / (I_high - I_low)
        if autoscale == "uint8":
            # convert it to uint8
            I = (I * 255).astype(np.uint8)
    return I



@app.post("/muscle", response_model=xxx)
def show_points(coords, labels, ax, marker_size=375):

    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



@app.post("/muscle", response_model=xxx)
def show_mask(mask, ax, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



@app.post("/muscle", response_model=xxx)
def empty_directory(directory):

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  #Remove files and links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)



@app.post("/muscle", response_model=xxx)
def show_anns(anns):

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



@app.post("/muscle", response_model=xxx)
def find_bounding_box(left_peak, right_peak, mask, mask_width):

  average_top = []
  average_bottom = []
  for i in range(left_peak, right_peak):
    top = 0
    while mask[top, i] == 0:
      top += 1
    average_top.append(top)
    bottom = top + mask_width[i]
    average_bottom.append(bottom)
  return sum(average_top)/len(average_top), sum(average_bottom)/len(average_bottom)



@app.post("/muscle", response_model=xxx)
def show_box(box, ax):

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

#Detecting bad mask:
#


@app.post("/muscle", response_model=xxx)
def bad_mask(j, mask_width, left_peak_to_edge, right_peak_to_edge):

  left_overflow = sum(mask_width[:j+15])
  right_overflow = sum(mask_width[len(mask_width)-16-j:])
  if (left_overflow > 0) or (right_overflow > 0):
    return True

  if left_peak_to_edge > 600 or right_peak_to_edge > 600 or left_peak_to_edge < 0 or right_peak_to_edge < 0:
    return True
  return False

#Find high oscillating area


@app.post("/muscle", response_model=xxx)
def oscillating(mask_width):

  window_size = 10
  smoothed_y = np.convolve(mask_width, np.ones(window_size)/window_size, mode='same')

  # Calculate local standard deviation
  window_size = 25  # You can adjust the window size
  local_std = np.array([np.std(mask_width[i:i+window_size]) for i in range(len(mask_width)-window_size+1)])
  local_std = np.pad(local_std, (window_size//2, window_size//2-1), mode='edge')

  # Detect abnormal oscillations
  abnormal_indices = np.where(local_std > 100)[0]
  return abnormal_indices




@app.post("/muscle", response_model=xxx)
def new_bad_mask(j, mask_width, left_peak_to_edge, right_peak_to_edge):

  abnormal = oscillating(mask_width)
  print(abnormal)
  if len(abnormal) > 0:
    return True
  return False

#Calculate left and right peak index, and calculate distance of left and right peak to image edge


@app.post("/muscle", response_model=xxx)
def calc_edge(mask_width):

  left = mask_width[:len(mask_width)//2]
  right = mask_width[len(mask_width)//2:]
  left_peak_index = mask_width.index(max(left))
  right_peak_index = len(left) + right.index(max(right))
  edge_left = 0
  edge_right = 0
  for i in range(len(mask_width)-1):
    if mask_width[0] != 0:
      edge_left = 0
      break
    elif mask_width[i] == 0 and mask_width[i+1] != 0:
      edge_left = i
      break
  for i in range(len(mask_width)-1, 0, -1):
    if mask_width[len(mask_width)-1] != 0:
      edge_right = len(mask_width)-1
      break
    elif mask_width[i] == 0 and mask_width[i-1] != 0:
      edge_right = i
      break
  return left_peak_index, right_peak_index, edge_left, edge_right, left_peak_index - edge_left, edge_right - right_peak_index



@app.post("/muscle", response_model=xxx)
def set_mask(image_rgb, j, input_box = None, input_points = None):

  mask_predictor.set_image(image_rgb)

  #Give Positive and/or Negative Point Prompt
  if input_points is None:
    input_point = np.array([[image_rgb.shape[1]//2, image_rgb.shape[0]//2], [j, image_rgb.shape[0]//2], [image_rgb.shape[1] - j, image_rgb.shape[0]//2]])
    input_label = np.array([1,0,0])
  else:
    input_point = np.array([[input_points[0], input_points[1]]])
    input_label = np.array([1])

  #Predict Mask from Point Prompts
  _, scores, logits = mask_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
  )
  mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
  masks, _, _ = mask_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    box = input_box,
    multimask_output=False,
  )
  return masks, scores, input_point, input_label



@app.post("/muscle", response_model=xxx)
def encode_image(image_array):

    #with open(filepath, 'rb') as f:
        #image_bytes = f.read()
    image = Image.fromarray(image_array)

    # Save the image to a buffer in JPEG format
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Read the image data from the buffer and encode it in Base64
    image_bytes = buffer.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded



@app.post("/muscle", response_model=xxx)
def merge_close_coords(coords, threshold=10):

    merged_coords = []
    used = np.zeros(len(coords), dtype=bool)

    for i in range(len(coords)):
        if used[i]:
            continue
        close_points = [coords[i]]
        used[i] = True

        for j in range(i + 1, len(coords)):
            if used[j]:
                continue
            if np.linalg.norm(np.array(coords[i]) - np.array(coords[j])) < threshold:
                close_points.append(coords[j])
                used[j] = True

        avg_point = np.mean(close_points, axis=0)
        merged_coords.append(list(avg_point))

    return merged_coords



@app.post("/muscle", response_model=xxx)
def find_contour_points(binary_mask):

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Extract contour points
    contour_points = []
    for contour in contours:
        for point in contour:
            contour_points.append(point[0])
    return contour_points



@app.post("/muscle", response_model=xxx)
def find_rough_contour(binary_mask):

  # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract contour points
    contour_points = []
    for contour in contours:
        for point in contour:
            contour_points.append(point[0])
    return contour_points



@app.post("/muscle", response_model=xxx)
def find_distance(point1, point2):

    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)



@app.post("/muscle", response_model=xxx)
def colorize_channel(channel, color):

    colored_image = np.zeros((channel.shape[0], channel.shape[1], 3), dtype=np.uint8)

    if color == 'magenta':
        # Magenta (R + B)
        colored_image[:, :, 0] = channel  # Red
        colored_image[:, :, 2] = channel  # Blue
    elif color == 'green':
        # Green
        colored_image[:, :, 1] = channel  # Green
    elif color == 'white':
        # White (R + G + B)
        colored_image[:, :, 0] = channel  # Red
        colored_image[:, :, 1] = channel  # Green
        colored_image[:, :, 2] = channel  # Blue

    return colored_image
from google.colab import drive
drive.mount('/content/drive')
base_directory = '/content/drive/MyDrive/Muscle_data'
import matplotlib.pyplot as plt
import cv2

#define a list each genotype subfolder contained in the larger folder
genotype = []
folder_name = 'msp300 Dys double het'
dfo = '/content/Muscle Analysis Template.xlsx'
genotype_name = folder_name
# Function to read and store images from a sub-folder


@app.post("/muscle", response_model=xxx)
def read_and_store_images(sub_folder, image_list):

    empty_directory(image_dir)
    folder_path = os.path.join(base_directory, sub_folder)
    filename = os.listdir(folder_path)
    sorted_filenames = sorted(filename, key=lambda x: (int(x.split()[1].split('.')[0]), int(x.split()[1].split('.')[1])))
    a_animal = [0]
    for i in range(1, len(sorted_filenames)):
        prev_animal = int(sorted_filenames[i-1].split()[1].split('.')[0])
        curr_animal = int(sorted_filenames[i].split()[1].split('.')[0])
        if prev_animal != curr_animal:
            a_animal.append(i * 2)

    for filename in sorted_filenames:
        if filename.endswith('.czi'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            img = AICSImage(file_path)
            channel_img = []
            for c in range(img.shape[1]):
              image = img.get_image_data("ZYX", C=c)
              image = np.max(image, axis = 0)
              channel_img.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            stacked_image = np.stack(channel_img, axis=-1)
            uint8_stacked = im_adjust(stacked_image, autoscale = 'uint8')


            image_list.append(uint8_stacked)
            base_filename = os.path.splitext(filename)[0]

            tiff_file_path = os.path.join(image_dir, f'{base_filename}.tiff')
            tifffile.imwrite(tiff_file_path, uint8_stacked)
    return a_animal
#Change name in parenthesis to the name of the genotype folder stored in google drive
#Spelling and capitalizations matter!
a_animal = read_and_store_images(folder_name, genotype)
from scipy.ndimage import binary_opening, binary_fill_holes, binary_closing

#set dystrophin to 1 if the image contain a dytrophin channel
dystrophin = 0

empty_directory(image_directory)
empty_directory(mask_dir)

j = 15
dir_name = f"n_point_xcoordinate:{j}"
full_directory = os.path.join(image_directory, dir_name)
os.makedirs(full_directory, exist_ok = True)
area = []
area_pixel = []
length = []
bad_image_index = []
contour = []
voronoi_contour = []
#Loop through each image in folder
for k, image in enumerate(genotype):

  bad_image = False
  image_rgb = cv2.cvtColor(image[:,:,dystrophin], cv2.COLOR_BGR2RGB)

  #Set initial mask
  masks, scores, input_point, input_label = set_mask(image_rgb, j)
  mask_width = []

  #Loop through each images for mask calculation
  for i, (mask, score) in enumerate(zip(masks, scores)):

  #Get mask width at each x-coord across the image
    mask = binary_closing(mask, np.ones([5,5]), iterations = 1)
    mask = binary_opening(mask, np.ones([5,5]), iterations = 1)
    binary_mask0 = mask > 0

    for verticals in range(0, image.shape[1]):
      mask_width.append(np.sum(binary_mask0[:, verticals]))

    #Calculating the peak width (marks the edges of the cell)
    left_peak, right_peak, edge_left, edge_right, left_peak_to_edge, right_peak_to_edge = calc_edge(mask_width)

    #Detecting bad mask
    box = None
    #if new_bad_mask(j, mask_width, left_peak_to_edge, right_peak_to_edge):
      #print('Bad Mask New!!!')
    if bad_mask(j, mask_width, left_peak_to_edge, right_peak_to_edge):
      print('Bad Mask')
      #Method 1: Simple threshold all mask width below 120 to 0
      for i in range(len(mask_width)):
        if mask_width[i] < 80:
          mask_width[i] = 0
          mask[:, i] = False
      __, __, edge_left, edge_right, new_left_peak_to_edge, new_right_peak_to_edge = calc_edge(mask_width)

      #Other possible methods:
      #move the negative points / add more positive points


      if bad_mask(j, mask_width, new_left_peak_to_edge, new_right_peak_to_edge):
        print('Please proceed to the next cell and manually draw bounding box for this image')
        #Method 3: Prompt user for manual input to set mask
        bad_image_index.append(k)
        bad_image = True

    #Calculate area in micron of mask
    binary_mask1 = mask > 0

    labeled_img, num_features = ndi.label(binary_mask1)
    size_threshold = 1000

    component_sizes = np.bincount(labeled_img.ravel())
    too_small = component_sizes < size_threshold
    too_small_mask = too_small[labeled_img]
    binary_mask1[too_small_mask] = 0

    np.save(os.path.join(mask_dir, f"mask_{k+1}.npy"), binary_mask1)
    mask_area_micron = np.sum(binary_mask1) / (2.4089**2)
    area.append(mask_area_micron)
    area.append('CELL')
    area_pixel.append(np.sum(binary_mask1))
    area_pixel.append('CELL')

    length_pixel = edge_right - edge_left
    length_micron = length_pixel / 2.4089
    length.append(length_micron)
    length.append('CELL')

    #find contour points for future analysis
    contour_points = find_contour_points(binary_mask1)
    contour.append(contour_points)
    rough_contour_points = find_rough_contour(binary_mask1)
    voronoi_contour.append(rough_contour_points)

    #Display Segmented result
    plt.figure(figsize=(10,10))
    plt.imshow(image_rgb)
    show_mask(binary_mask1, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, N point coordinate: {j}", fontsize=18)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    image_name = f"image_{k+1}.png"
    plt.savefig(os.path.join(full_directory, image_name), bbox_inches='tight', pad_inches=0)
    plt.show()
    print('Image index:', k)

area.pop()
area_pixel.pop()
length.pop()
IS_COLAB = True

if IS_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()
from jupyter_bbox_widget import BBoxWidget

#Change the current_index each time if multiple images need manual segmentation
bad_mask_index = 3
print('You can use your mouse to click on the image to insert positive points and/or draw bounding box')
bad_image_rgb = cv2.cvtColor(genotype[bad_mask_index][:,:,0+dystrophin], cv2.COLOR_BGR2RGB)
widget = BBoxWidget()
widget.image = encode_image(bad_image_rgb)
widget
positive_point = None
box = None
if len(widget.bboxes) > 1:
  if widget.bboxes[0]['width'] == 0:
    box = widget.bboxes[1]
    box = np.array([
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ])
    positive_point = widget.bboxes[0]
    positive_point = np.array([
        positive_point['x'],
        positive_point['y']
    ])
  else:
    box = widget.bboxes[0]
    box = np.array([
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ])
    positive_point = widget.bboxes[1]
    positive_point = np.array([
        positive_point['x'],
        positive_point['y']
    ])
else:
  if widget.bboxes[0]['width'] == 0:
    positive_point = widget.bboxes[0]
    positive_point = np.array([
        positive_point['x'],
        positive_point['y']
    ])
  else:
    box = widget.bboxes[0]
    box = np.array([
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ])

if box is None:
  box_point_mask, __, points, point_label = set_mask(bad_image_rgb, j, input_points = positive_point)

elif positive_point is None:
  box_point_mask, __, points, point_label = set_mask(bad_image_rgb, j, box)

else:
  box_point_mask, __, points, point_label = set_mask(bad_image_rgb, j, box, positive_point)

binary_mask1 = box_point_mask[0] > 0

binary_mask1 = binary_closing(binary_mask1, np.ones([5,5]), iterations = 1)
binary_mask1 = binary_opening(binary_mask1, np.ones([5,5]), iterations = 1)

labeled_img, num_features = ndi.label(binary_mask1)
size_threshold = 5000

component_sizes = np.bincount(labeled_img.ravel())
too_small = component_sizes < size_threshold
too_small_mask = too_small[labeled_img]
binary_mask1[too_small_mask] = 0

#Display Segmented result
plt.figure(figsize=(10,10))
plt.imshow(bad_image_rgb)
show_mask(binary_mask1, plt.gca())
if positive_point is not None:
  show_points(points, point_label, plt.gca())
#plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#image_name = f"image_{k+1}.png"
#plt.savefig(os.path.join(full_directory, image_name), bbox_inches='tight', pad_inches=0)
if box is not None:
  show_box(box, plt.gca())
plt.show()
file_path = os.path.join(mask_dir, f"mask_{bad_mask_index+1}.npy")
os.remove(file_path)
np.save(os.path.join(mask_dir, f"mask_{bad_mask_index+1}.npy"), binary_mask1)
mask_area_micron = np.sum(binary_mask1) / (2.4089**2)
area[bad_mask_index*2] = mask_area_micron
new_contour = find_contour_points(binary_mask1)
contour[bad_mask_index] = new_contour
new_voronoi_contour = find_rough_contour(binary_mask1)
voronoi_contour[bad_mask_index] = new_voronoi_contour



@app.post("/muscle", response_model=xxx)
def display_image_z(sub_folder, nuclei):

    folder_path = os.path.join(base_directory, sub_folder)
    filename = os.listdir(folder_path)
    sorted_filenames = sorted(filename, key=lambda x: (int(x.split()[1].split('.')[0]), int(x.split()[1].split('.')[1])))

    for filename in sorted_filenames:
        if filename.endswith('.czi'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            img = AICSImage(file_path)
            channel_img = []
            #Change the percentage of z-planes to be stacked
            percentage = 0.8
            planes = int(img.shape[2] * percentage)
            for z in range(planes):
              image = img.get_image_data('YX', Z = z, C = 2+dystrophin)
              image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
              image = im_adjust(image, autoscale = 'uint8')
              plt.imshow(image)
              plt.show()
            for c in range(img.shape[1]):
              image = img.get_image_data("ZYX", C=c)
              image = image[:planes]
              image = np.max(image, axis = 0)
              channel_img.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            stacked_image = np.stack(channel_img, axis=-1)
            uint8_stacked = im_adjust(stacked_image, autoscale = 'uint8')
            nuclei.append(uint8_stacked[:,:,2])
nuclei_list = []
display_image_z(folder_name, nuclei_list)
#Three parameters that you may need to change:
pixel_threshold = 3
size_threshold = 650
gaussian_blur = 15
optima_window_size = 45

nuclei = []
x = []
y = []
x_pixel = []
y_pixel = []
distance_to_edge = []
closest_edge = []
n_animal = []
x_animal = []
nuclei_num = []
n_animal.append(0)
x_animal.append(0)
nuclei_num.append(0)

for k, image in enumerate(nuclei_list):
  nuclei.append(1)
  nuclei.append(2)
  mask_path = os.path.join(mask_dir, f"mask_{k+1}.npy")
  loaded_mask = np.load(mask_path)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  masked_image = image_rgb * loaded_mask[..., None]
  masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
  print('Image', k+1)
  plt.figure(figsize=(20,20))
  plt.subplot(121)
  plt.imshow(image_rgb)
  plt.subplot(122)
  plt.imshow(masked_image)
  plt.show()

  blurred = cv2.GaussianBlur(masked_image,(gaussian_blur, gaussian_blur),0)

  bin_img = blurred > pixel_threshold
  #bin_img = binary_opening(bin_img, np.ones((5, 5)))
  bin_img = binary_fill_holes(bin_img)
  labeled_img, num_features = ndi.label(bin_img)

  component_sizes = np.bincount(labeled_img.ravel())
  too_small = component_sizes < size_threshold
  too_small_mask = too_small[labeled_img]
  bin_img[too_small_mask] = 0
  plt.figure(figsize=(20,20))
  plt.imshow(bin_img)
  plt.show()

  distance = ndi.distance_transform_edt(bin_img)

  coords = peak_local_max(distance, footprint=np.ones((optima_window_size, optima_window_size)), labels=bin_img)
  ycoord, xcoord = zip(*coords)
  merged_coords = merge_close_coords(list(zip(ycoord, xcoord)), threshold=10)
  merged_coords = [(int(x), int(y)) for x, y in merged_coords]  # Ensure integer coordinates
  ycoords, xcoords = zip(*merged_coords)

  #Calculate closest nuclei to edge distance

  closest_edges = []
  distance_to_edges = []
  for i in range(len(xcoords)):
    min_distance = float('inf')
    closest_point = None
    for border in contour[k]:
      d = find_distance([xcoords[i], ycoords[i]], border)
      if d < min_distance:
        min_distance = d
        closest_point = border

    closest_edges.append(closest_point)
    distance_micron = min_distance / 2.4089
    distance_to_edges.append(distance_micron)

  closest_edge.extend(closest_edges)
  distance_to_edge.extend(distance_to_edges)

  #convert x y coordinates to micron
  micron_x = []
  micron_y = []
  pixel_x = []
  pixel_y = []
  for i in range(len(xcoords)):
    micron_x.append(xcoords[i] / 2.4089)
    micron_y.append(ycoords[i] / 2.4089)
    pixel_x.append(xcoords[i])
    pixel_y.append(ycoords[i])
  x.extend(micron_x)
  y.extend(micron_y)
  x_pixel.extend(pixel_x)
  y_pixel.extend(pixel_y)

  # Create markers for watershed
  mask = np.zeros(distance.shape, dtype=bool)
  mask[tuple(np.array(merged_coords).T)] = True
  markers, _ = ndi.label(mask)

  # Apply watershed algorithm
  seg_watershed = watershed(-distance, markers, mask=bin_img)

  plt.figure(figsize=(20,20))
  plt.imshow(seg_watershed)
  plt.scatter(xcoords, ycoords, c='red', s=5)
  plt.show()
  nuclei.extend(range(3, 3+len(merged_coords)))

  nuclei.append('CELL')
  x.append('CELL')
  y.append('CELL')
  x_pixel.append('CELL')
  y_pixel.append('CELL')
  closest_edge.append('CELL')
  distance_to_edge.append('CELL')
  nuclei_num.append(len(closest_edge))
  for i in range(1, len(a_animal)):
    if (k+1)*2 == a_animal[i]:
      n_animal.append(len(nuclei))
      x_animal.append(len(x))
nuclei.pop()
x.pop()
y.pop()
distance_to_edge.pop()
closest_edge.pop()
x_pixel.pop()
y_pixel.pop()
for i in range(len(nuclei_num)-1):
  plt.figure(figsize=(20,20))
  for k in range(nuclei_num[i], nuclei_num[i+1] - 1):
      plt.imshow(nuclei_list[i], cmap = 'gray')
      plt.plot([x_pixel[k], closest_edge[k][0]], [y_pixel[k], closest_edge[k][1]], linestyle='-', linewidth=5, color='blue')
      plt.scatter([x_pixel[k], closest_edge[k][0]], [y_pixel[k], closest_edge[k][1]], color='red', s = 100)
      x_contour = [points[0] for points in contour[i]]
      y_contour = [points[1] for points in contour[i]]
      plt.plot(x_contour, y_contour, linestyle='-', linewidth=2, color='red')
      plt.plot()
  plt.show()
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from scipy.ndimage.morphology import binary_erosion
from sklearn.neighbors import KDTree
from os.path import exists
import matplotlib.patches as patches
import matplotlib.path as mpath
import random
from shapely.geometry import MultiPolygon



@app.post("/muscle", response_model=xxx)
def voronoi_finite_polygons_2d(vor, boundary=None):

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if boundary is None:
        radius = vor.points.ptp().max() * 10
    else:
        radius = np.max(np.linalg.norm(boundary - center, axis=1)) * 10

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
polygon_area = []
empty_directory(voronoi_dir)
animal_index = []
for num in a_animal:
  animal_index.append(num/2)
which_animal = 1
for i in range(len(nuclei_num) - 1):
    if which_animal < len(animal_index):
      if i == animal_index[which_animal]:
        which_animal += 1
    which_cell = i-animal_index[which_animal-1]+1
    nuclei_points = [[x, y] for x, y in zip(x_pixel[nuclei_num[i]: nuclei_num[i + 1] - 1],
                                            y_pixel[nuclei_num[i]: nuclei_num[i + 1]])]
    nuclei_points = np.array(nuclei_points)
    border_points = voronoi_contour[i]
    boundary = Polygon(border_points)
    vor = Voronoi(nuclei_points)
    regions, vertices = voronoi_finite_polygons_2d(vor, border_points)

    fig, ax = plt.subplots()

    boundary_patch = patches.Polygon(np.array(boundary.exterior.coords), closed=True, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(boundary_patch)

    # Plot the clipped Voronoi regions with different colors

    for region in regions:
        polygon = vertices[region]
        poly = Polygon(polygon)
        poly = poly.intersection(boundary)
        if isinstance(poly, MultiPolygon):
          print('multiPolygon')
          for sub_poly in poly.geoms:  # Access individual polygons with .geoms
            if not sub_poly.is_empty:
                poly_area = sub_poly.area / (2.4089**2)
                polygon_area.append(poly_area)
                # Generate a random color
                color = [random.random(), random.random(), random.random()]
                patch = patches.Polygon(np.array(sub_poly.exterior.coords), edgecolor='blue', facecolor=color, alpha=0.4)
                ax.add_patch(patch)
        else:
          if not poly.is_empty:
              poly_area = poly.area / (2.4089**2)
              polygon_area.append(poly_area)
              # Generate a random color
              color = [random.random(), random.random(), random.random()]
              patch = patches.Polygon(np.array(poly.exterior.coords), edgecolor='blue', facecolor=color, alpha=0.4)
              ax.add_patch(patch)

    polygon_area.append("CELL")
    # Plot the points
    ax.plot(nuclei_points[:, 0], nuclei_points[:, 1], 'ro', markersize=4)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Overlay the Voronoi diagram on the original images
    grayscale_image = cv2.cvtColor(genotype[i], cv2.COLOR_BGR2GRAY)
    ax.imshow(grayscale_image, cmap='gray', extent=(0, genotype[i].shape[1], 0, genotype[i].shape[0]), origin='lower')
    plt.xlim(0, genotype[i].shape[1])
    plt.ylim(genotype[i].shape[0], 0)

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(voronoi_dir, f'Voronoi {int(which_animal)}.{int(which_cell)}.jpg'), format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
print(polygon_area)
shutil.make_archive('/content/voronoi_diagrams', 'zip', voronoi_dir)
files.download('/content/voronoi_diagrams.zip')
n_animal.append(len(nuclei) + 1)
a_animal.append(len(area) + 1)
x_animal.append(len(x) + 1)


@app.post("/muscle", response_model=xxx)
def main(wb_name, ws_name):

    #reading in and formatting lists of data from the input file
    #finding indices of every 'ANIMAL' delimeter in the lists so we can
    #pass in the correct number of muscles at a time

    #passing in chunks of data into the MuscleAnalysis.py script
    #passes data from index 0 to first 'ANIMAL' delimeter, then from first to second 'ANIMAL' delimeter, and so on

    for i in range(len(n_animal) - 1):
      ma.main(nuclei[n_animal[i] : n_animal[i + 1] - 1],
              area[a_animal[i] : a_animal[i + 1] - 1],
              x[x_animal[i] : x_animal[i + 1] - 1],
              y[x_animal[i] : x_animal[i + 1] - 1],
              length[a_animal[i] : a_animal[i + 1] - 1],
              distance_to_edge[x_animal[i] : x_animal[i + 1] - 1],
              polygon_area[x_animal[i] : x_animal[i + 1] - 1],
              wb_name, ws_name, i)

    #writing data and calculations to excel output file
    wb = load_workbook(wb_name)
    try:
      ws = wb['Sheet']
      ws.title = 'Summary'
    except:
      ws = wb.create_sheet('Summary')
    ws['A1'] = 'Muscle'
    ws['B1'] = 'Number of Nuclei'
    ws['C1'] = 'Ave Internuclear Distance'
    ws['D1'] = 'Muscle Length'
    ws['E1'] = 'Muscle Area'
    ws['F1'] = 'Ave A:I Ratio'
    ws['G1'] = 'Voronoi Area COV'
    ws['H1'] = 'Nuclei to Edge Distance'

    nuclei_nums = []
    ave_inter_nuc_dists = []
    mus_lens = []
    mus_areas = []
    ave_AI_rat = []
    ave_voronoi_cov = []
    ave_nuclei_to_edge = []

    c_row = 2
    for sheet in wb:
        if sheet.title == 'Summary':
            continue
        break_points = []
        for i in range(len(sheet['A'])):
            val = str(sheet['A' + str(i + 1)].value)
            if 'Muscle:' in val:
                break_points.append(i + 1)
        for val in break_points:
            ws['A' + str(c_row)] = sheet['A' + str(val)].value[7:]
            if sheet['B' + str(val)] == 'NO DATA':
                ws['B' + str(c_row)] = 'NO DATA'
                c_row += 1
                continue
            ws['B' + str(c_row)] = sheet['B' + str(val + 1)].value
            nuclei_nums.append(sheet['B' + str(val + 1)].value)

            ws['C' + str(c_row)] = sheet['D' + str(val + 1)].value
            ave_inter_nuc_dists.append(sheet['D' + str(val + 1)].value)

            ws['D' + str(c_row)] = sheet['F' + str(val + 1)].value
            mus_lens.append(sheet['F' + str(val + 1)].value)

            ws['E' + str(c_row)] = sheet['G' + str(val + 1)].value
            mus_areas.append(sheet['G' + str(val + 1)].value)

            ws['F' + str(c_row)] = sheet['I' + str(val + 1)].value
            ave_AI_rat.append(sheet['I' + str(val + 1)].value)

            ws['G' + str(c_row)] = sheet['K' + str(val + 1)].value
            ave_voronoi_cov.append(sheet['K' + str(val + 1)].value)

            ws['H' + str(c_row)] = sheet['O' + str(val + 1)].value
            ave_nuclei_to_edge.append(sheet['O' + str(val + 1)].value)
            c_row += 1
    wb.save(wb_name)
    return nuclei_nums, ave_inter_nuc_dists, mus_lens, mus_areas, ave_AI_rat

#formatting an output CSV file with the calculated data to be passed into R (or into the code blocks below)
num, dist, lens, areas, ratio = main(dfo, genotype_name)
df = pd.DataFrame({'Genotype': [genotype_name for x in range(len(dist))],
                  'NumNuclei': num,
                  'InternuclearDistance': dist,
                  'MuscleLength': lens,
                  'MuscleArea': areas,
                  'A:IRatio': ratio})
df.to_csv(genotype_name + '-RInput.csv', encoding = 'utf-8-sig')
