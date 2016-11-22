import argparse
import yaml

import pypeliner

import remixt.mappability.bwa.workflow


def run(**args):
    ref_data_dir = args['ref_data_dir']

    config = yaml.load(open(args['config']))

    pypeliner_config = config.copy()
    pypeliner_config.update(args)
    pyp = pypeliner.app.Pypeline(config=pypeliner_config)

    workflow = remixt.mappability.bwa.workflow.create_bwa_mappability_workflow(config, ref_data_dir)

    pyp.run(workflow)
    

def add_arguments(argparser):
    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    argparser.set_defaults(func=run)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
