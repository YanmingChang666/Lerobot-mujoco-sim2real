"""Build a fixed-base writing scene from the actual T-pose URDF, not x2cT.xml."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'robots/x02c_tpose/assets'
JOINT_NAMES = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
               'right_shoulder_yaw_joint', 'right_elbow_joint']


def origin(element):
    o = element.find('origin')
    if o is None: return {}
    xyzw = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_quat()
    return {'pos': o.get('xyz', '0 0 0'), 'quat': ' '.join(map(str, xyzw[[3,0,1,2]]))}


def main():
    source = ET.parse(ASSETS/'x2c_TPose.urdf').getroot()
    links = {l.get('name'): l for l in source.findall('link')}
    children = {}
    child_names = set()
    for j in source.findall('joint'):
        children.setdefault(j.find('parent').get('link'), []).append(j)
        child_names.add(j.find('child').get('link'))
    root_link = next(name for name in links if name not in child_names)
    model = ET.Element('mujoco', model='x02cTpose_right_writing')
    ET.SubElement(model,'compiler',angle='radian',meshdir='meshes')
    ET.SubElement(model,'option',timestep='.002',integrator='implicitfast')
    default=ET.SubElement(model,'default')
    ET.SubElement(default,'joint',damping='.2',armature='.02',frictionloss='0')
    ET.SubElement(default,'geom',contype='0',conaffinity='0')
    assets=ET.SubElement(model,'asset'); world=ET.SubElement(model,'worldbody')
    ET.SubElement(world,'light',pos='1 -2 3',dir='-1 1 -2',castshadow='false')
    ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .01',rgba='.75 .78 .8 1')
    mesh_names={}
    def build(name,parent,attrs,joint=None):
        body=ET.SubElement(parent,'body',name=name,gravcomp='1',**attrs)
        link=links[name]; inertial=link.find('inertial')
        if inertial is not None and float(inertial.find('mass').get('value'))>0:
            inertia=inertial.find('inertia'); a=origin(inertial)
            # fullinertia is specified in body axes; rotate URDF inertia explicitly.
            q=np.fromstring(a.pop('quat','1 0 0 0'),sep=' ')
            r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            vals={k:float(inertia.get(k)) for k in ['ixx','iyy','izz','ixy','ixz','iyz']}
            I=np.array([[vals['ixx'],vals['ixy'],vals['ixz']],[vals['ixy'],vals['iyy'],vals['iyz']],[vals['ixz'],vals['iyz'],vals['izz']]])
            I=r@I@r.T
            ET.SubElement(body,'inertial',mass=inertial.find('mass').get('value'),fullinertia=' '.join(map(str,[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])),**a)
        if joint is not None and joint.get('name') in JOINT_NAMES:
            lim=joint.find('limit')
            ET.SubElement(body,'joint',name=joint.get('name'),axis=joint.find('axis').get('xyz'),range=f"{lim.get('lower')} {lim.get('upper')}")
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None: continue
            file=Path(mesh.get('filename')).name
            if file not in mesh_names:
                mesh_names[file]='mesh_'+str(len(mesh_names))
                ET.SubElement(assets,'mesh',name=mesh_names[file],file=file,scale=mesh.get('scale','1 1 1'))
            color=visual.find('material/color')
            ET.SubElement(body,'geom',type='mesh',mesh=mesh_names[file],rgba=color.get('rgba') if color is not None else '.8 .8 .8 1',**origin(visual))
        for j in children.get(name,[]):build(j.find('child').get('link'),body,origin(j),j)
    build(root_link,world,{'pos':'0 0 1.042'})
    wrist=model.find('.//body[@name="right_wrist_roll_link"]')
    ET.SubElement(wrist,'geom',name='pen',type='capsule',fromto='.1 0 0 .1 0 -.16',size='.003',rgba='.12 .12 .12 1',mass='.01')
    ET.SubElement(wrist,'site',name='pen_tip',pos='.1 0 -.16',quat='0 0 1 0',size='.003',rgba='0 0 1 1')
    ET.SubElement(world,'geom',name='paper',type='box',pos='.3 -.25 .848',size='.13 .13 .002',rgba='.97 .97 .94 1')
    actuator=ET.SubElement(model,'actuator')
    for name in JOINT_NAMES:
        joint=next(j for j in source.findall('joint') if j.get('name')==name)
        effort=float(joint.find('limit').get('effort'))
        ET.SubElement(actuator,'velocity',name=name,joint=name,kv='60',ctrlrange='-.5 .5',forcerange=f'{-effort} {effort}')
    ET.SubElement(ET.SubElement(model,'keyframe'),'key',name='home',qpos='0 0 0 0',ctrl='0 0 0 0')
    ET.indent(model,space='  ')
    ET.ElementTree(model).write(ASSETS/'right_write.xml',encoding='unicode',xml_declaration=True)


if __name__=='__main__':main()
