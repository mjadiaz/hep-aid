#from hepread import *
import re 

slha_file = 'Users/madiaz/WorkArea/BLSSM_model_files/BLSSM_work/ScanBLSSM_HBHS/SPhenoBLSSM_output/SPheno.spc.BLSSM_HEscan_0' 
work_dir = 'Users/madiaz/WorkArea/BLSSM_model_files/BLSSM_work'

pattern = r'(.+)(?P<value>.\d+\.\d+E.\d+)\s+(?P<comment>#.*)'
decay_body_pattern = r'\s+(?P<value>.\d+\.\d+E.\d+)(?P<entries>.+)\s+(?P<comment>#.*)'

linet1 = '2312 01050105 3232   00   0    -0.84656259E-15    # coeffBB_SRRSM'
linet2 = '6.03026606E-01    2            2   -1000024   # BR(Sd_6 -> Fu_1 Cha_1 )'
decay_header= r'DECAY\s+(?P<particle>\w+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)'

decaybodyline = '   1.35614064E-03    2           22         22   # BR(hh_2 -> VP VP )'

decayline = 'DECAY        35     1.27115040E-03   # hh_2'
match = re.match(decay_body_pattern, decaybodyline)
entries = match.group('entries').split()
value = match.group('value')
comment = match.group('comment')

print(entries, value, comment)
#entries_format = '{:4s} ' * (len(entries)-1) + '{:6s} ' + '{:20s} ' + '{}'  
#print(entries_format.format(*(entries + [value] + [comment])))
#print(match.group('comment'))
#print(match.group(2))
#print(match.group(1).split(),len(match.group(1).split()))
