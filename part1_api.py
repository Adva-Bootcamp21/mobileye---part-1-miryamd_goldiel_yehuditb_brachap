# import pyvips as pyvips

try:
    import test
    import os
    import json
    import glob
    import argparse
    import cv2

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from scipy import misc
    from PIL import Image
    import matplotlib.pyplot as plt
    from scipy.spatial import distance

except ImportError:
    print("Need to fix the installation")
    raise


def build_kernel():
    circle_img = Image.open("light.png").convert('L')
    kernel = np.asarray(circle_img)
    kernel = kernel.astype(np.float32)
    kernel -= 100
    sum_circle = np.sum(kernel)
    area = circle_img.width * circle_img.height
    kernel -= (sum_circle / area)
    max_kernel = np.max(kernel)
    kernel /= max_kernel
    return kernel


def find_tfl_lights(c_image: np.ndarray, kernel, some_threshold):
    red_matrix, green_matrix = np.array(c_image)[:, :, 0], np.array(c_image)[:, :, 1]

    new_red = sg.convolve(red_matrix, kernel, mode='same')
    new_green = sg.convolve(green_matrix, kernel, mode='same')

    red_max = maximum_filter(new_red, size=250)
    green_max = maximum_filter(new_green, size=250)
    red_max_point = red_max == new_red
    green_max_point = green_max == new_green

    y_red, x_red = np.where(red_max_point)
    y_green, x_green = np.where(green_max_point)

    # Remove the dot with a lower value rgb between two point  in 20 pixels
    for i in range(len(x_red)):
        for j in range(len(x_green)):
            if x_red[i] == -1 or y_red[i] == -1 or x_green[j] == -1 or y_green[j] == -1:
                D = distance.euclidean((x_red[i], y_red[i]), (x_green[j], y_green[j]))
                dist = math.sqrt((x_red[i] - y_red[i]) ** 2 + (x_green[j] - y_green[j]) ** 2)
                if dist < 50:
                    if green_max[x_red[i]][y_red[i]] > red_max[x_green[j]][y_green[j]]:
                        x_red[i] = -1
                        y_red[i] = -1
                        break
                    else:
                        x_green[j] = -1
                        y_green[j] = -1
                        break

    for index in range(len(x_red)):
        if y_red[index] < 30 or y_red[index] > 1014 :
            x_red[index] = -1
            y_red[index] = -1
    y_red = np.delete(y_red, np.where( y_red==-1))
    x_red = np.delete(x_red,np.where(x_red==-1))

    for index in range(len(x_green)):
        if y_green[index] < 20 or y_green[index] > 1019:
            x_green[index] = -1
            y_green[index] = -1

    y_green = np.delete(y_green,np.where(y_green==-1))
    x_green = np.delete(x_green,np.where(x_green ==-1))

    return x_red, y_red, x_green, y_green


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, statistics, kernel, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    img = cv2.imread(image_path)
    small_img = cv2.pyrDown(img)

    red_x_small, red_y_small, green_x_small, green_y_small = find_tfl_lights(small_img, kernel, some_threshold=42)
    image = np.array(Image.open(image_path))

    red_x_big, red_y_big, green_x_big, green_y_big = find_tfl_lights(image, kernel, some_threshold=42)

    a1 = np.array(red_x_small * 2)
    a2 = np.array(red_y_small * 2)
    b1 = np.array(red_x_big)
    b2 = np.array(red_y_big)
    c1 = np.array(green_x_small * 2)
    c2 = np.array(green_y_small * 2)
    d1 = np.array(green_x_big)
    d2 = np.array(green_y_big)

    red_x = np.concatenate([a1, b1])
    red_y = np.concatenate([a2, b2])
    green_x = np.concatenate([c1, d1])
    green_y = np.concatenate([c2, d2])

    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
        polygons = []
        for o in objects:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            polygons.append(poly)
        update_statistics(polygons, red_x, red_y, green_x, green_y, statistics)

    show_image_and_gt(image, objects, fig_num)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def update_statistics(polygons, red_x, red_y, green_x, green_y, statistics):
    points = []
    for i in range(len(red_x)):
        points.append((red_x[i], red_y[i]))
    for i in range(len(green_x)):
        points.append((green_x[i], green_y[i]))
    num_marked_points = 0
    for polygon in polygons:
        for point in points:
            if (test.is_inside_polygon(polygon, point)):
                statistics[0]['traffic_light'] += 1
                num_marked_points += 1
                break
    statistics[0]['traffic_without_light'] += len(polygons) - num_marked_points
    statistics[0]['light_without_traffic'] += len(points) - num_marked_points


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'test_data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    statistic = [{'traffic_light': 0, 'traffic_without_light': 0, 'light_without_traffic': 0}]
    load = 0
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        kernel = build_kernel()
        test_find_tfl_lights(image, statistic, kernel, json_fn)

        load += 1
        print(load, len(flist))
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    print(statistic)
    plt.show(block=True)


if __name__ == '__main__':
    main()