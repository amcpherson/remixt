import argparse
import itertools

import utils


def interface(Tool):

    parser = argparse.ArgumentParser()

    parser.add_argument('install_directory', help='Installation directory')

    subparsers = parser.add_subparsers(dest='command')

    install_parser = subparsers.add_parser('install')
    
    init_parser = subparsers.add_parser('init')
    
    init_parser.add_argument('analysis_directory',
        help='Analysis directory')

    init_parser.add_argument('normal_filename',
        help='Input normal filename')

    init_parser.add_argument('tumour_filename',
        help='Input tumour filename')

    init_parser.add_argument('--segment_filename', required=False,
        help='Input segments filename')

    init_parser.add_argument('--perfect_segment_filename', required=False,
        help='Input perfect segments filename')

    init_parser.add_argument('--breakpoint_filename', required=False,
        help='Input breakpoints filename')

    init_parser.add_argument('--haplotype_filename', required=False,
        help='Input haplotypes filename')

    run_parser = subparsers.add_parser('run')

    run_parser.add_argument('analysis_directory',
        help='Analysis directory')

    run_parser.add_argument('--init_param_idx', type=int, default=None,
        help='Input Initialization params index')

    report_parser = subparsers.add_parser('report')

    report_parser.add_argument('analysis_directory',
        help='Analysis directory')

    report_parser.add_argument('output_cn_filename',
        help='Results output filename')

    report_parser.add_argument('output_mix_filename',
        help='Results output filename')

    args = parser.parse_args()

    tool = Tool(args.install_directory)

    if args.command == 'install':

        tool.install()

    elif args.command == 'prepare':

        analysis = tool.create_analysis(args.analysis_directory)

        num_init = analysis.prepare(
            args.normal_filename,
            args.tumour_filename,
            segment_filename=args.segment_filename,
            perfect_segment_filename=args.perfect_segment_filename,
            breakpoint_filename=args.breakpoint_filename,
            haplotype_filename=args.haplotype_filename,
        )

        print ('Number of initializations = {0}'.format(num_init))

    elif args.command == 'run':

        analysis = tool.create_analysis(args.analysis_directory)

        if args.init_param_idx is None:
            for init_param_idx in itertools.count():
                try:
                    analysis.run(init_param_idx)
                except utils.InvalidInitParam:
                    break
        else:
            analysis.run(args.init_param_idx)

    elif args.command == 'report':

        analysis = tool.create_analysis(args.analysis_directory)

        analysis.report(
            args.output_cn_filename,
            args.output_mix_filename,
        )





