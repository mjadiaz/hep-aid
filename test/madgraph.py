
def test_read_mg_generation_info():
    from hepaid.hepread import read_mg_generation_info
    path = '/scratch/mjad1g20/HEP/MG5_aMC_v3_5_0/testee/Events/run_01/run_01_tag_1_banner.txt'
    #path = '/scratch/mjad1g20/HEP/MG5_aMC_v3_5_0/SLHA_BLSSM/pph1aa/Events/run_01/run_01_tag_1_banner.txt'
    lines = read_mg_generation_info(path)
    print(lines)



if __name__ == '__main__':
    pass