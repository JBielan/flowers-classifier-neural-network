import argparse
import json
import my_functions as mf

parser = argparse.ArgumentParser()

parser.add_argument('img_dir', type=str, default='/home/workspace/paind-project/flowers/test/65/image_03243.jpg')

parser.add_argument('chpt', type=str, default='checkpoint_vgg13.pth')

parser.add_argument('--top_k', action='store', 
                    dest='top_k', default='5')

parser.add_argument('--category_names', action='store', 
                    dest='map_name', default='')

parser.add_argument('--gpu', action='store_const',
                    dest='device',
                    const='cuda',
                   default='cpu')

# Mapping class names in the dictionary
args = parser.parse_args()
label_map = {}
if args.map_name:
    with open(args.map_name, 'r') as f:
        label_map = json.load(f)
      
# Loading choosen model    
model = mf.load_model(args.chpt)

top_p, top_flowers = mf.predict(args.img_dir, model, label_map, args.top_k)
print(top_p)
print(top_flowers)