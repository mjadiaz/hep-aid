from omegaconf import OmegaConf

from heptools import Spheno 
from hepread import LesHouches
import subprocess
import os

cfg = OmegaConf.load('hep_tools.yaml')

lhef = LesHouches(	
		file_dir=cfg.reference_lhs,
		work_dir = cfg.work_dir,
		model=cfg.model.name
		)
print(lhef.block_list)

lhef.block('MINPAR').show()
lhef.block('YXIN').show()
for line in lhef.block('EXTPAR').block_body:
	print(line.options)

for block_name in lhef.model_param_blocks():
	lhef.block(block_name).show()
