import argparse
import itertools

import utils


def interface(Wrapper):

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

    wrapper = Wrapper(args.install_directory)

    if args.command == 'install':

        wrapper.install()

    if args.command == 'init':

        init_params = wrapper.init(
            args.analysis_directory,
            args.normal_filename,
            args.tumour_filename,
            segment_filename=args.segment_filename,
            perfect_segment_filename=args.perfect_segment_filename,
            breakpoint_filename=args.breakpoint_filename,
            haplotype_filename=args.haplotype_filename,
        )

        print init_params

    elif args.command == 'run':

        if args.init_param_idx is None:

            for init_param_idx in itertools.count():

                try:

                    wrapper.run(
                        args.analysis_directory,
                        init_param_idx,
                    )

                except utils.InvalidInitParam:
                    break

        else:

            wrapper.run(
                args.analysis_directory,
                args.init_param_idx,
            )

    elif args.command == 'report':

        wrapper.report(
            args.analysis_directory,
            args.output_cn_filename,
            args.output_mix_filename,
        )





