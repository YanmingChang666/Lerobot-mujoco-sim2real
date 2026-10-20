"""Create a right-arm writing scene from the supplied a07 MJCF; leave source intact."""
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def main():
    tree = ET.parse(HERE / 'assets/urdf/a07.xml')
    root = tree.getroot()
    root.set('model', 'X02_right_arm_writing')
    root.find('compiler').set('meshdir', 'meshes')
    root.remove(root.find('default'))
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'joint', damping='.2', armature='.02', frictionloss='0')
    ET.SubElement(root, 'option', timestep='.002', integrator='implicitfast')
    # Keep torso and left arm visible, but weld the left arm at its source zero pose.
    for body in root.findall('.//body'):
        body.set('gravcomp', '1')
        if body.get('name', '').endswith('_l'):
            for joint in body.findall('joint'):
                body.remove(joint)
    # Source contains duplicate visual/collision meshes; retain only one per body.
    for parent in root.iter():
        seen = set()
        for geom in list(parent.findall('geom')):
            mesh = geom.get('mesh')
            if mesh in seen:
                parent.remove(geom)
                continue
            seen.add(mesh)
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
    root.remove(root.find('actuator'))
    actuator = ET.SubElement(root, 'actuator')
    for i in range(1, 8):
        ET.SubElement(actuator, 'velocity', name=f'joint{i}_r', joint=f'joint{i}_r',
                      kv='60', ctrlrange='-.5 .5', forcerange='-40 40')
    wrist = root.find('.//body[@name="link7_r"]')
    ET.SubElement(wrist, 'geom', name='pen', type='capsule',
                  fromto='0 0 -.04 0 0 -.16', size='.003', mass='.01',
                  rgba='.12 .12 .12 1', contype='0', conaffinity='0')
    # Pen extends down local -Z; site +Z follows the pen, with X reversed.
    ET.SubElement(wrist, 'site', name='pen_tip', pos='0 0 -.16', quat='0 0 1 0',
                  size='.003', rgba='0 0 1 1')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='0 -1 2', dir='0 0 -1', directional='true', castshadow='false')
    ET.SubElement(world, 'geom', name='paper', type='box', pos='.25 -.23 -.002',
                  size='.13 .13 .002', rgba='.97 .97 .94 1', contype='0', conaffinity='0')
    ET.SubElement(root, 'keyframe')
    ET.SubElement(root.find('keyframe'), 'key', name='home', qpos='0 0 0 0 0 0 0', ctrl='0 0 0 0 0 0 0')
    ET.indent(tree, space='  ')
    output = HERE / 'assets/x02_right_write.xml'
    tree.write(output, encoding='unicode', xml_declaration=True)
    print(output)


if __name__ == '__main__':
    main()
