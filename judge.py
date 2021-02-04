import os
import subprocess
import math
# 数据集的放置位置
dataset_location = "./input/DataSet"
output_path = "./judge_output.json"


def get_lane(theta, radius):
    # double cost = cos(((double)theta / 180.0) * M_PI), sint = sin(((double)theta / 180.0) * M_PI);
    # *input.GetPoint(i, y) = print_val; int y = (radius - i * cost) / sint;
    row = 160
    lane = "["
    while True:
        if(row >= 720):
            break
        col = (radius-row*math.cos((theta/180.0)*math.pi)) / \
            (math.sin(math.pi*theta/180.0))
        if col < 0 or col >= 1280:
            col = -2
        col = int(col)
        lane += str(col)
        if row < 710:
            lane += ", "
        row += 10
    lane += "]"
    return lane


if __name__ == '__main__':
    f = open(output_path, "w+")
    for root, dirs, files in os.walk(dataset_location):
        for file in files:
            if "20.jpg" in file:
                image_path = root+'/'+file
                print(image_path)
                json = '{"lanes": ['

                # value = os.system("./DipFantasy %s" % image_path)
                value = subprocess.run(
                    ["./DipFantasy", image_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
                # print(value)
                line = value.split('\n')
                line_num = 0
                for paraments in line:
                    if ' ' in paraments:
                        if line_num != 0:
                            json += ", "
                        line_num += 1
                        theta, radius = paraments.split(' ')
                        print("theta:"+theta)
                        print("radius:"+radius)
                        lane = get_lane(theta=float(theta),
                                        radius=float(radius))
                        # print(lane)
                        json += lane
                json += '], "h_samples": ['
                row = 160
                while True:
                    json += str(row)
                    row += 10
                    if row >= 720:
                        break
                    json += ', '

                json += '], "raw_file": "'
                json += image_path.replace(dataset_location, "clips")
                json += '" ,"run_time":0}'

                # print(image_path.replace(dataset_location, "clips/"))
                f.writelines(json+'\n')
                print(json)

    os.system("python3 ./judge/lane.py %s ./judge/groundtruth.json"%output_path)