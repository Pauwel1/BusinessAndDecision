import xml.etree.ElementTree as ET 

# Pass the path of the xml document 
tree = ET.parse('/Users/paww/Documents/GitHub/BusinessAndDecision/data/train_xml/0a948131fe85c38152c0b9b22f7c09fc_3.xml') 

# get the parent tag 
root = tree.getroot() 

# print the root (parent) tag along with its memory location 
print("The root attributes: ", root) 

# print the attributes of the first tag  
print("First tag: ", root[0].attrib) 

# print the text contained within first subtag of the 5th tag from the parent 
print(root.text) 