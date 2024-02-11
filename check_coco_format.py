from pycocotools.coco import COCO
import json

def check_coco_format(json_file_path):
    # 加载JSON文件
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    # 创建COCO实例
    coco = COCO()
    coco.dataset = coco_data
    coco.createIndex()

    # 验证格式
    ann_ids = coco.getAnnIds()
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()

    # 检查是否存在图像、注释和类别
    if not img_ids:
        print("错误：缺少图像数据。")
        return False
    if not ann_ids:
        print("错误：缺少注释数据。")
        return False
    if not cat_ids:
        print("错误：缺少类别数据。")
        return False

    # 验证注释中的图像ID和类别ID是否有效
    for ann_id in ann_ids:
        ann = coco.loadAnns(ann_id)[0]
        if ann['image_id'] not in img_ids:
            print(f"错误：注释 {ann_id} 中的图像ID无效。")
            return False
        if ann['category_id'] not in cat_ids:
            print(f"错误：注释 {ann_id} 中的类别ID无效。")
            return False

    print("COCO格式验证通过。")
    return True

# 在此处替换为你的coco_instances.json文件路径
json_file_path = '/ssd3/wyy/projects/mmdetection/data/ESD_COCO_instances/annotations/instances_train2017.json'

# 检查coco_instances.json文件的格式
check_coco_format(json_file_path)
