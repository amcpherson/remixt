import argparse
import yaml
import os

import pypeliner
import pypeliner.managed as mgd

import biowrappers.pipelines.setup_reference_dbs

import remixt.simulations.pipeline


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('tool_defs',
        help='Tool Definition Filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    yaml_text = open(args['tool_defs']).read().format(ref_data_dir=args['ref_data_dir'])
    tool_defs = yaml.load(yaml_text)
    databases = tool_defs['databases']
    tool_defs = tool_defs['tools']

    if 'tmpdir' not in args:
        args['tmpdir'] = os.path.join(args['ref_data_dir'], 'setup_tmp')

    pyp = pypeliner.app.Pypeline(config=args)

    workflow = biowrappers.pipelines.setup_reference_dbs.create_setup_reference_dbs_workflow(databases)

    pyp.run(workflow)

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(
        obj=mgd.TempOutputObj('tool_defs', 'tool_name'),
        value=tool_defs,
    )

    workflow.subworkflow(
        name='setup_tool',
        axes=('tool_name',),
        func=remixt.simulations.pipeline.run_setup_function,
        args=(
            mgd.TempInputObj('tool_defs', 'tool_name'),
            databases,
        ),
    )

    pyp.run(workflow)

