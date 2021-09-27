# [HiggsBounds GitLab](https://gitlab.com/higgsbounds/higgsbounds)

- [Manual](https://arxiv.org/pdf/2006.06007.pdf)


HiggsBounds takes a selection of Higgs sector predictions for any particular model as input and then uses the experimental topological cross section limits from Higgs searches at LEP, the Tevatron and the LHC to determine if this parameter point has been excluded at 95% C.L..

**Compilation**

HiggsBounds requires a Fortran compiler supporting at least Fortran 95 (like `gfortran`) and `cmake`.

The code is compiled by

```bash
mkdir build && cd build
cmake ..
make
```

### HiggsBound in Sarah and SPheno
- [Sarah GitLab](https://gitlab.in2p3.fr/goodsell/sarah/-/wikis/HiggsBounds)