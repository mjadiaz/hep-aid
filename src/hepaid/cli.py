import click
from hepaid.hep.install import install_hepstack
from hepaid.hep.install import write_config_template

@click.group()
def cli():
    """HEP Aid Command Line Interface."""
    pass

@cli.command()
@click.option('--spheno_url', '-s', default="https://spheno.hepforge.org/downloads?f=SPheno-4.0.4.tar.gz", help='URL for the Spheno installation.')
@click.option('--spheno_compiler', '-c', default='gfortran', help='Compiler for Spheno.')
@click.option('--spheno_on_mac', '-m', default=True, type=bool, help='Is Spheno being installed on MacOS?')
@click.option('--spheno_model_dir', default="BLSSM_SPheno", help='Directory for the Spheno model.')
@click.option('--spheno_model_name', default="BLSSM", help='Name of the Spheno model.')
@click.option('--higgsbounds_url', default='https://gitlab.com/higgsbounds/higgsbounds/-/archive/master/higgsbounds-master.tar.gz', help='URL for HiggsBounds.')
@click.option('--higgssignals_url', default='https://gitlab.com/higgsbounds/higgssignals/-/archive/master/higgssignals-master.tar.gz', help='URL for HiggsSignals.')
@click.option('--madgraph_url', default="https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.4.tar.gz", help='URL for MadGraph installation.')
@click.option('--madgraph_model_dir', default='BLSSM_UFO', help='Directory for the MadGraph model.')
@click.option('--madgraph_model_name', default='BLSSM', help='Name of the MadGraph model.')
def install_hepstack_cli(spheno_url, spheno_compiler, spheno_on_mac, spheno_model_dir, spheno_model_name, higgsbounds_url, higgssignals_url, madgraph_url, madgraph_model_dir, madgraph_model_name):
    """
    Install the full SARAH Family HEPStack: Spheno, HiggsBounds, HiggsSignals, and MadGraph.
    """
    install_hepstack(
        spheno_url=spheno_url,
        spheno_compiler=spheno_compiler,
        spheno_on_mac=spheno_on_mac,
        spheno_model_dir=spheno_model_dir,
        spheno_model_name=spheno_model_name,
        higgsbounds_url=higgsbounds_url,
        higgssignals_url=higgssignals_url,
        madgraph_url=madgraph_url,
        madgraph_model_dir=madgraph_model_dir,
        madgraph_model_name=madgraph_model_name
    )


@cli.command()
@click.option('--spheno-dir', default='', help='Directory for Spheno.')
@click.option('--hb-dir', default='', help='Directory for HiggsBounds.')
@click.option('--hs-dir', default='', help='Directory for HiggsSignals.')
@click.option('--mg-dir', default='', help='Directory for MadGraph.')
def generate_config_template(spheno_dir, hb_dir, hs_dir, mg_dir):
    """Generate a configuration template."""
    write_config_template(spheno_dir, hb_dir, hs_dir, mg_dir)
    
if __name__ == '__main__':
    cli()