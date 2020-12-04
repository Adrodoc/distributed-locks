from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['savefig.transparent'] = True


upb_group_by_rank = True
upb_reorder_cohort_locks = False
ecsb_throughput_legend_loc = None
ccwb_28_latency_max_ylim = None
def max_wbab_latency(suffix): return 8


def read_benchmark_json(path):
    # print('Reading: ', path)
    with open(path) as file:
        content = json.load(file)
    runs_df = pd.json_normalize(content['runs'])
    context_df = pd.json_normalize([content['context']])
    return runs_df.merge(context_df, how='cross')


def sort_and_rename_locks(data):
    global max_wbab_latency
    global upb_group_by_rank
    global upb_reorder_cohort_locks
    global ecsb_throughput_legend_loc
    global ccwb_28_latency_max_ylim
    name_mapping = None

    # Baseline:
    # def max_wbab_latency(suffix): return 8 if 'processes=112' in suffix else 4
    # name_mapping = {
    #     'MpiWinLock': 'MPI',
    #     'DashLock': 'DASH',
    #     'DMcsLock': 'D-MCS',
    #     'RmaMcsLock': 'RMA-MCS',
    # }

    # D-MCS Optimierung:
    # def max_wbab_latency(suffix): return 8
    # name_mapping = {
    #     'DMcsLock': 'D-MCS',
    #     'McsLockAccumulate': '+ MPI_Win_flush',
    #     'McsLock': '+ MPI_Put',
    #     'McsLockAtomic': '+ std::atomic',
    # }

    # DASH Optimierung:
    # def max_wbab_latency(suffix): return 8
    # name_mapping = {
    #     'DashLock': 'DASH',
    #     'McsLockDashStyle': 'Reimplementierung',
    #     'McsLockDashStyleCheckNextInRelease': '+ Nachfolger prüfen',
    #     'McsLockDashStyleDirectSpinning': '+ Direkte Zugriffe',
    #     'McsLockTwoSided': '+ MPI_Put',
    #     'McsLockTwoSidedAtomic': '+ std::atomic',
    # }

    # Baseline optimiert:
    # def max_wbab_latency(suffix): return 8
    # name_mapping = {
    #     'MpiWinLock': 'MPI',
    #     'McsLockTwoSided': 'DASH (MPI_Put)',
    #     'McsLock': 'D-MCS (MPI_Put)',
    #     'RmaMcsLock': 'RMA-MCS',
    #     'McsLockTwoSidedAtomic': 'DASH (atomic)',
    #     'McsLockAtomic': 'D-MCS (atomic)',
    # }

    # RH UPB:
    # data.drop(data.loc[data['benchmark'] != 'UPB'].index, inplace=True)
    # upb_group_by_rank = False
    # name_mapping = {
    #     'RhLock-local_max=32': 'RH',
    #     'McsLock': 'D-MCS (MPI_Put)',
    #     'RmaMcsLock': 'RMA-MCS',
    # }

    # RH non UPB:
    # data.drop(data.loc[data['benchmark'] == 'UPB'].index, inplace=True)
    # def max_wbab_latency(suffix): return 8 if 'processes=112' in suffix else 4
    # name_mapping = {
    #     'RhLock-local_max=16': 'RH (lbx=16, f=1)',
    #     'RhLock-local_max=32': 'RH (lbx=32, f=1)',
    #     'RhLock-local_max=32-fair_factor=2': 'RH (lbx=32, f=2)',
    #     'RhLock-local_max=32-fair_factor=100': 'RH (lbx=32, f=100)',
    #     'RhLock-local_max=64': 'RH (lbx=64, f=1)',
    #     'McsLock': 'D-MCS (MPI_Put)',
    #     'RmaMcsLock': 'RMA-MCS',
    # }

    # HCLH:
    # def max_wbab_latency(suffix): return 32
    # name_mapping = {
    #     'ClhLock': 'CLH',
    #     'ClhLockNuma': 'CLH (NUMA)',
    #     # 'McsLockMpi': 'MCS',
    #     'McsLock': 'D-MCS (MPI_Put)',
    # }

    # TAS vs TTS:
    # name_mapping = {
    #     'TasLock': 'TAS',
    #     'TasLockBo': 'TAS-BO',
    #     'TasLockCas': 'TAS-CAS',
    #     'TasLockCasBo': 'TAS-CAS-BO',
    #     'TtsLock': 'TTS',
    #     'TtsLockBo': 'TTS-BO',
    #     'TtsLockCas': 'TTS-CAS',
    #     'TtsLockCasBo': 'TTS-CAS-BO',
    # }

    # Cohort counter optimization:
    # def max_wbab_latency(suffix): return 4 if 'processes=112' in suffix else 2
    # name_mapping = {
    #     'RmaMcsLock': 'RMA-MCS',
    #     'CohortLock_McsLockTwoSided_McsLockAtomicWithCohortDetection': 'C-MCS-MCS',
    #     'CohortLockLocalCounter_McsLockTwoSided_McsLockAtomicWithCohortDetection': '+ lokaler Zähler',
    #     'CohortLockDirectCounter_McsLockTwoSided_McsLockAtomicWithCohortDetection': '+ direkter Zähler',
    #     'CohortLockInlineCounter_McsLockTwoSided_McsLockAtomicWithCohortDetection': '+ Inline-Zähler',
    #     # 'CohortLockInlineCounter_McsLock_McsLockAtomicWithCohortDetection': '- P2P',
    # }

    # Cohort inline:
    # def max_wbab_latency(suffix): return 4 if 'processes=112' in suffix else 2
    # upb_reorder_cohort_locks = True
    # ecsb_throughput_legend_loc = 'lower left'
    # name_mapping = {
    #     'CohortLockInlineCounter_TasLockCasBo_McsLockAtomicWithCohortDetection': 'C-TAS-MCS',
    #     'CohortLockInlineCounter_McsLockTwoSided_McsLockAtomicWithCohortDetection': 'C-MCS-MCS',
    #     'CohortLockInlineCounter_TktLock_McsLockAtomicWithCohortDetection': 'C-TKT-MCS',
    #     'CohortLockInlineCounter_TasLockCasBo_TktLockAtomic': 'C-TAS-TKT',
    #     'CohortLockInlineCounter_McsLockTwoSided_TktLockAtomic': 'C-MCS-TKT',
    #     'CohortLockInlineCounter_TktLock_TktLockAtomic': 'C-TKT-TKT',
    #     'CohortLockInlineCounter_TasLockCasBo_TtsLockBoAtomicWithCohortDetection': 'C-TAS-TTS',
    #     'CohortLockInlineCounter_McsLockTwoSided_TtsLockBoAtomicWithCohortDetection': 'C-MCS-TTS',
    #     'CohortLockInlineCounter_TktLock_TtsLockBoAtomicWithCohortDetection': 'C-TKT-TTS',
    # }

    # Cohort inline hem:
    # def max_wbab_latency(suffix): return 1
    # ccwb_28_latency_max_ylim = (2.375, 2.7)
    # name_mapping = {
    #     'CohortLockInlineCounter_McsLockTwoSided_HemLockAtomic': 'C-MCS-HEM',
    #     'CohortLockInlineCounter_McsLockTwoSided_HemLockOverlapAtomic': 'C-MCS-HEM (Overlap)',
    #     'CohortLockInlineCounter_McsLockTwoSided_HemLockCtrAtomic': 'C-MCS-HEM (CTR)',
    #     # 'CohortLockInlineCounter_McsLockTwoSided_HemLockCtrOverlapAtomic': 'C-MCS-HEM (CTR & Overlap)',
    #     'CohortLockInlineCounter_McsLockTwoSided_HemLockCtrAhAtomic': 'C-MCS-HEM (CTR & AH)',
    #     'CohortLockInlineCounter_McsLockTwoSided_HemLockCtrAhAtomicWithCohortDetection': 'C-MCS-HEM (CTR & AH & inline)',
    #     'CohortLockInlineCounter_McsLockTwoSided_ClhLockAtomicWithCohortDetection': 'C-MCS-CLH',
    #     'CohortLockInlineCounter_McsLockTwoSided_McsLockAtomicWithCohortDetection': 'C-MCS-MCS',
    # }

    # ShflLock:
    # def max_wbab_latency(suffix): return 32 if 'processes=112' in suffix else 4
    # name_mapping = {
    #     'ShflLock': 'SHFL',
    #     'ShflLockGlobalTas': 'SHFL (mit TAS)',
    #     'McsLockWithTtsStealing': 'MCS (mit TTS)',
    #     'McsLockWithTasStealing': 'MCS (mit TAS)',
    #     'McsLock': 'MCS',
    # }

    if name_mapping is not None:
        data['lock'] = pd.Categorical(
            data['lock'], categories=name_mapping.keys(), ordered=True)
        data.sort_values('lock', inplace=True)
        data['lock'] = data['lock'].map(name_mapping)
        data.dropna(subset=['lock'], inplace=True)


plot_dir = Path(__file__).resolve().parent
reports_dir = plot_dir / '../reports'
for commit_path in reports_dir.iterdir():
    commit = commit_path.name
    json_dir = commit_path / 'json'
    png_dir = commit_path / 'png'
    if not json_dir.is_dir():
        print('Skipping '+commit)
        continue
    if png_dir.is_dir():
        print('Skipping '+commit)
        continue
    print('Plotting '+commit)

    data = pd.concat([
        read_benchmark_json(p)
        for p in sorted(json_dir.iterdir())
        if p.suffix == '.json'
    ]).reset_index()

    sort_and_rename_locks(data)

    png_dir.mkdir(exist_ok=True)

    if 'critical_work' in data.columns:
        data['critical_work'] = data['critical_work']\
            .fillna(-1).astype('int64')
    if 'avg_wait_ns' in data.columns:
        data['avg_wait_ns'] = pd.to_numeric(data['avg_wait_ns'])
        data['avg_wait_us'] = data['avg_wait_ns'] / 1000
    if 'processes' in data.columns:
        data['processes'] = data['processes'].fillna(-1).astype('int64')

    # data = data[(data['benchmark'] == 'WBAB') & (data['processes'] == 112)]

    benchmarks = []
    for (benchmark, df) in data.groupby('benchmark'):
        if benchmark == 'UPB':
            for (lock_count, df) in df.groupby('lock_count'):
                benchmarks.append(
                    (benchmark, '-lock_count='+str(lock_count), df))
        elif benchmark == 'WBAB':
            for (processes, df) in df.groupby('processes'):
                for (mpi_progress, df) in df.groupby('mpi_progress'):
                    benchmarks.append(
                        (benchmark, '-processes='+str(processes)+',mpi_progress='+mpi_progress, df))
        elif benchmark == 'CCWB':
            for (processes, df) in df.groupby('processes'):
                benchmarks.append(
                    (benchmark, '-processes='+str(processes), df))
        else:
            benchmarks.append((benchmark, '', df))

    benchmarks.reverse()
    for (benchmark, suffix, df) in benchmarks:
        if df['lock'].dtype.name == 'category':
            df['lock'] = df['lock'].cat.remove_unused_categories()

        if benchmark == 'UPB':
            df = df.rename(columns={
                'same_process_master_rank_ns': '1a',
                'same_node_master_rank_ns': '2a',
                'different_node_master_rank_ns': '3a',
                'same_process_master_node_ns': '1b',
                'same_node_master_node_ns': '2b',
                'different_node_master_node_ns': '3b',
                'same_process_slave_node_ns': '1c',
                'same_node_slave_node_ns': '2c',
                'different_node_slave_node_ns': '3c',
            })
            df = df.melt(var_name='scenario', value_name='duration_ns', id_vars='lock', value_vars=[
                '1a',
                '2a',
                '3a',
                '1b',
                '2b',
                '3b',
                '1c',
                '2c',
                '3c',
            ])
            df['previous_owner'] = df['scenario'].str.get(0)\
                .map({'1': 'Selber Prozess', '2': 'Selber Knoten', '3': 'Anderer Knoten'})
            df['rank'] = df['scenario'].str.get(-1)\
                .map({'a': 'Hauptprozess', 'b': 'auf Hauptknoten', 'c': 'auf entferntem Knoten'})

            x_axis = 'rank' if upb_group_by_rank else 'scenario'
            hue_order = None
            palette = None
            if upb_reorder_cohort_locks:
                hue_order = [
                    'C-TAS-MCS',
                    'C-TAS-TKT',
                    'C-TAS-TTS',
                    'C-MCS-MCS',
                    'C-MCS-TKT',
                    'C-MCS-TTS',
                    'C-TKT-MCS',
                    'C-TKT-TKT',
                    'C-TKT-TTS',
                ]
                palette = ['C0', 'C3', 'C6', 'C1',
                           'C4', 'C7', 'C2', 'C5', 'C8']

            df['throughput'] = 1000 / df['duration_ns']
            plot = sns.barplot(
                data=df,
                estimator=np.median,
                x=x_axis, y='throughput',
                hue='lock',
                capsize=.05,
                hue_order=hue_order,
                palette=palette,
            )
            plot.set(
                ylim=0,
                ylabel='Durchsatz in μs\nMedian mit 95 % Konfidenzintervall',
                xlabel='Akquirierender Prozess' if upb_group_by_rank else 'Szenario',
            )
            plot.grid(axis='y')
            plot.legend(title='Lock')
            fig = plot.get_figure()
            fig.savefig(png_dir / (benchmark+suffix+'-throughput.png'))
            fig.clf()

            df['latency'] = df['duration_ns'] / 1000
            plot = sns.barplot(
                data=df,
                estimator=np.median,
                x=x_axis, y='latency',
                hue='lock',
                capsize=.05,
                hue_order=hue_order,
                palette=palette,
            )
            plot.set(
                ylim=0,
                ylabel='Iterationsdauer in μs\nMedian mit 95 % Konfidenzintervall',
                xlabel='Akquirierender Prozess' if upb_group_by_rank else 'Szenario',
            )
            plot.grid(axis='y')
            plot.legend(title='Lock')
            fig = plot.get_figure()
            fig.savefig(png_dir / (benchmark+suffix+'-latency.png'))
            fig.clf()
            continue

        x_axis = 'critical_work' if benchmark == 'CCWB' else 'avg_wait_us' if benchmark == 'WBAB' else 'processes'
        xlabel = 'Kritische Arbeit' if benchmark == 'CCWB' else 'Durchschn. Wartezeit in μs pro Prozess' if benchmark == 'WBAB' else 'Prozesse'

        df['throughput'] = df['iterations'] * 1000 / df['duration_ns']
        plot = sns.lineplot(
            data=df,
            estimator=np.median,
            x=x_axis, y='throughput',
            hue='lock', style='lock', markers=True, dashes=False,
        )
        plot.set(
            ylim=None if benchmark == 'CCWB' else 0,
            ylabel='Durchsatz in Mio/s\nMedian mit 95 % Konfidenzintervall',
            xlabel=xlabel,
        )
        if benchmark == 'WBAB':
            plot.set_xscale('symlog', base=2, linscale=0.5, linthresh=0.25)
            plot.get_xaxis().set_major_formatter('{x:g}')
        if x_axis == 'processes':
            plot.tick_params(axis='x', which='minor', length=3.5)
            plot.set_xticks(
                df[x_axis].drop_duplicates().sort_values().iloc[1::2], minor=False)
            plot.set_xticks(df[x_axis].drop_duplicates(
            ).sort_values().iloc[::2], minor=True)
            plot.get_xaxis().set_minor_formatter('   {x}')
            plot.get_xaxis().set_major_formatter('{x}   ')
            plot.axvline(29, color='k', alpha=.5, linestyle='dashed', zorder=0)
            plot.axvline(56.5, color='k', alpha=.5,
                         linestyle='dashed', zorder=0)
            plot.axvline(86, color='k', alpha=.5, linestyle='dashed', zorder=0)
            plot.grid(axis='y')
        else:
            plot.set_xticks(df[x_axis].drop_duplicates())
            plot.grid()
        if benchmark == 'WBAB':
            x = df[x_axis].drop_duplicates().sort_values()
            y = 1/x
            plot.plot(x, y, label='Optimum',
                      color='k', alpha=.5, linestyle='dashed', zorder=0)
        if benchmark == 'ECSB' and ecsb_throughput_legend_loc is not None:
            plot.legend(title='Lock', loc=ecsb_throughput_legend_loc)
        else:
            plot.legend(title='Lock')
        fig = plot.get_figure()
        fig.savefig(png_dir / (benchmark+suffix+'-throughput.png'))
        fig.clf()

        df['latency'] = df['duration_ns'] / df['iterations'] / 1000
        limit = max_wbab_latency(suffix)
        latency_df = df if benchmark != 'WBAB' else df[df[x_axis] <= limit]
        plot = sns.lineplot(
            data=latency_df,
            estimator=np.median,
            x=x_axis, y='latency',
            hue='lock', style='lock', markers=True, dashes=False,
        )
        plot.set(
            ylim=None if benchmark == 'CCWB' else 0,
            ylabel='Iterationsdauer in μs\nMedian mit 95 % Konfidenzintervall',
            xlabel=xlabel,
        )
        if x_axis == 'processes':
            plot.tick_params(axis='x', which='minor', length=3.5)
            plot.set_xticks(
                latency_df[x_axis].drop_duplicates().sort_values().iloc[1::2], minor=False)
            plot.set_xticks(latency_df[x_axis].drop_duplicates(
            ).sort_values().iloc[::2], minor=True)
            plot.get_xaxis().set_minor_formatter('   {x}')
            plot.get_xaxis().set_major_formatter('{x}   ')
            plot.axvline(29, color='k', alpha=.5, linestyle='dashed', zorder=0)
            plot.axvline(56.5, color='k', alpha=.5,
                         linestyle='dashed', zorder=0)
            plot.axvline(86, color='k', alpha=.5, linestyle='dashed', zorder=0)
            plot.grid(axis='y')
        else:
            plot.set_xticks(latency_df[x_axis].drop_duplicates().sort_values())
            plot.grid()
        if benchmark == 'WBAB':
            if limit > 8:
                plot.set_xticks(np.delete(plot.get_xticks(), [1, 2]))
            plot.get_xaxis().set_major_formatter('{x:g}')
            x = latency_df[x_axis].drop_duplicates().sort_values()
            y = x
            plot.plot(x, y, label='Optimum',
                      color='k', alpha=.5, linestyle='dashed', zorder=0)
        plot.legend(title='Lock')
        fig = plot.get_figure()
        fig.savefig(png_dir / (benchmark+suffix+'-latency.png'))
        fig.clf()

        if benchmark == 'CCWB':
            max_df = df[df[x_axis] == df[x_axis].max()]
            plot = sns.barplot(
                data=max_df,
                estimator=np.median,
                x=x_axis, y='latency',
                hue='lock',
                capsize=.05,
            )
            plot.set(
                ylim=ccwb_28_latency_max_ylim
                if ccwb_28_latency_max_ylim is not None and 'processes=28' in suffix
                else (max_df['latency'].min(), max_df['latency'].max()),
                ylabel='Iterationsdauer in μs\nMedian mit 95 % Konfidenzintervall',
                xlabel=xlabel,
            )
            plot.grid(axis='y')
            plot.legend(title='Lock')
            fig = plot.get_figure()
            fig.savefig(png_dir / (benchmark+suffix+'-latency-max.png'))
            fig.clf()

        if benchmark == 'WBAB':
            df['overhead'] = df['latency'] - df['avg_wait_us']
            plot = sns.lineplot(
                data=df,
                estimator=np.median,
                x=x_axis, y='overhead',
                hue='lock', style='lock', markers=True, dashes=False,
                # palette=['C0', 'C0', 'C0', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2']
                # palette=['C0', 'C1', 'C2', 'C0', 'C1',
                #          'C2', 'grey', 'grey', 'grey']
            )
            plot.set(
                ylim=(0, 1),
                xlim=-0.1,
                ylabel='Overhead in μs\nMedian mit 95 % Konfidenzintervall',
                xlabel=xlabel,
            )
            plot.set_xscale('symlog', base=2, linscale=0.5, linthresh=0.25)
            plot.get_xaxis().set_major_formatter('{x:g}')
            plot.set_xticks(df[x_axis].drop_duplicates())
            plot.grid()
            plot.legend(title='Lock')

            # Annotate low overhead
            if False:
                from matplotlib.patches import Ellipse
                from matplotlib.transforms import ScaledTranslation

                c = 'k'

                point = (0.25, 0.375)
                ell_offset = ScaledTranslation(
                    point[0], point[1], plot.transScale)
                ell_tform = ell_offset + plot.transLimits + plot.transAxes
                plot.add_patch(Ellipse(xy=(0, 0), width=0.22, height=0.8,
                                       angle=55, edgecolor=c, fc='None', lw=1, transform=ell_tform))
                plot.annotate('C-*-MCS', xy=point,
                              xytext=(-50, -65), textcoords='offset points',
                              color=c, size='large',
                              arrowprops=dict(
                                  arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                  facecolor=c, shrinkB=30))

                point = (0.275, 0.7)
                ell_offset = ScaledTranslation(
                    point[0], point[1], plot.transScale)
                ell_tform = ell_offset + plot.transLimits + plot.transAxes
                plot.add_patch(Ellipse(xy=(0, 0), width=0.2, height=1.1,
                                       angle=40, edgecolor=c, fc='None', lw=1, transform=ell_tform))
                plot.annotate('C-*-TKT', xy=point,
                              xytext=(10, 60), textcoords='offset points',
                              color=c, size='large',
                              arrowprops=dict(
                                  arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                  facecolor=c, shrinkB=30))

                point = (2.75, 0.9)
                ell_offset = ScaledTranslation(
                    point[0], point[1], plot.transScale)
                ell_tform = ell_offset + plot.transLimits + plot.transAxes
                plot.add_patch(Ellipse(xy=(0, 0), width=0.5, height=1.5,
                                       angle=20, edgecolor=c, fc='None', lw=1, transform=ell_tform))
                plot.annotate('C-*-TTS', xy=point,
                              xytext=(-90, -80), textcoords='offset points',
                              color=c, size='large',
                              arrowprops=dict(
                                  arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                  facecolor=c, shrinkB=60))

            # Annotate medium overhead
            if False:
                circle_rad = 7

                px = 1
                point = (px, df[(df['lock'] == 'C-MCS-MCS') &
                                (df[x_axis] == px)]['overhead'].median())
                plot.plot(point[0], point[1], 'o', ms=circle_rad * 2,
                          mfc='none', mec=c, mew=1)
                plot.annotate('C-MCS-*', xy=point,
                              xytext=(-90, -20), textcoords='offset points',
                              color=c, size='large',
                              arrowprops=dict(
                                  arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                  facecolor=c, shrinkB=circle_rad * 1.5))

                px = 2
                point = (px, df[(df['lock'] == 'C-TKT-TKT') &
                                (df[x_axis] == px)]['overhead'].median())
                plot.plot(point[0], point[1], 'o', ms=circle_rad * 2,
                          mfc='none', mec=c, mew=1)
                plot.annotate('C-TKT-*', xy=point,
                              xytext=(-30, 45), textcoords='offset points',
                              color=c, size='large',
                              arrowprops=dict(
                                  arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                  facecolor=c, shrinkB=circle_rad * 1.5))

                px = 4
                point = (px, df[(df['lock'] == 'C-TAS-MCS') &
                                (df[x_axis] == px)]['overhead'].median())
                plot.plot(point[0], point[1], 'o', ms=circle_rad * 2,
                          mfc='none', mec=c, mew=1)
                plot.annotate('C-TAS-*', xy=point,
                              xytext=(-85, -20), textcoords='offset points',
                              color=c, size='large',
                              arrowprops=dict(
                                  arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                  facecolor=c, shrinkB=circle_rad * 1.5))

            fig = plot.get_figure()
            fig.savefig(png_dir / (benchmark+suffix+'-overhead.png'))
            fig.clf()

        df['iterations_per_process_cv'] = df['iterations_per_process_cv'] * 100
        plot = sns.lineplot(
            data=df,
            estimator=np.median,
            x=x_axis, y='iterations_per_process_cv',
            hue='lock', style='lock', markers=True, dashes=False,
        )
        plot.set(
            # ylim=0,
            ylabel='CV des Prozessfortschritts in %\nMedian mit 95 % Konfidenzintervall',
            xlabel=xlabel,
        )
        if benchmark == 'WBAB':
            plot.set_xscale('symlog', base=2, linscale=0.5, linthresh=0.25)
            plot.get_xaxis().set_major_formatter('{x:g}')
        if x_axis == 'processes':
            plot.tick_params(axis='x', which='minor', length=3.5)
            plot.set_xticks(
                df[x_axis].drop_duplicates().sort_values().iloc[1::2], minor=False)
            plot.set_xticks(df[x_axis].drop_duplicates(
            ).sort_values().iloc[::2], minor=True)
            plot.get_xaxis().set_minor_formatter('   {x}')
            plot.get_xaxis().set_major_formatter('{x}   ')
            plot.axvline(29, color='k', alpha=.5, linestyle='dashed', zorder=0)
            plot.axvline(56.5, color='k', alpha=.5,
                         linestyle='dashed', zorder=0)
            plot.axvline(86, color='k', alpha=.5, linestyle='dashed', zorder=0)
            plot.grid(axis='y')
        else:
            plot.set_xticks(df[x_axis].drop_duplicates())
            plot.grid()
        plot.legend(title='Lock')
        fig = plot.get_figure()
        fig.savefig(png_dir / (benchmark+suffix+'-fairness.png'))
        fig.clf()

        if 'stats.local_release_cnt' in df.columns:
            df['local_releases'] = df['stats.local_release_cnt'] / \
                (df['stats.global_release_cnt'] +
                 df['stats.local_release_cnt']) * 100
            plot = sns.lineplot(
                data=df,
                estimator=np.median,
                x=x_axis, y='local_releases',
                hue='lock', style='lock', markers=True, dashes=False,
            )
            plot.set(
                ylabel='Anteil lokaler Lockübergaben in %\nMedian mit 95 % Konfidenzintervall',
                xlabel=xlabel,
            )
            if benchmark == 'WBAB':
                plot.set_xscale('symlog', base=2, linscale=0.5, linthresh=0.25)
                plot.get_xaxis().set_major_formatter('{x:g}')
            if x_axis == 'processes':
                plot.tick_params(axis='x', which='minor', length=3.5)
                plot.set_xticks(
                    df[x_axis].drop_duplicates().sort_values().iloc[1::2], minor=False)
                plot.set_xticks(df[x_axis].drop_duplicates(
                ).sort_values().iloc[::2], minor=True)
                plot.get_xaxis().set_minor_formatter('   {x}')
                plot.get_xaxis().set_major_formatter('{x}   ')
                plot.axvline(29, color='k', alpha=.5,
                             linestyle='dashed', zorder=0)
                plot.axvline(56.5, color='k', alpha=.5,
                             linestyle='dashed', zorder=0)
                plot.axvline(86, color='k', alpha=.5,
                             linestyle='dashed', zorder=0)
                plot.grid(axis='y')
            else:
                plot.set_xticks(df[x_axis].drop_duplicates())
                plot.grid()
            plot.legend(title='Lock')
            fig = plot.get_figure()
            fig.savefig(png_dir / (benchmark+suffix + '-local-releases.png'))
            fig.clf()

        for stats_prefix in ['', 'global.', 'local.']:
            acquired_delayed = 'stats.'+stats_prefix+'acquired_delayed'
            if acquired_delayed not in df.columns:
                continue
            df['contention'] = df[acquired_delayed] / \
                (df['stats.'+stats_prefix+'acquired_immediately'] +
                 df[acquired_delayed]) * 100
            plot = sns.lineplot(
                data=df,
                estimator=np.median,
                x=x_axis, y='contention',
                hue='lock', style='lock', markers=True, dashes=False,
            )
            plot.set(
                ylabel='Konkurrenz in %\nMedian mit 95 % Konfidenzintervall',
                xlabel=xlabel,
            )
            if benchmark == 'WBAB':
                plot.set_xscale('symlog', base=2, linscale=0.5, linthresh=0.25)
                plot.get_xaxis().set_major_formatter('{x:g}')
            if x_axis == 'processes':
                plot.tick_params(axis='x', which='minor', length=3.5)
                plot.set_xticks(
                    df[x_axis].drop_duplicates().sort_values().iloc[1::2], minor=False)
                plot.set_xticks(df[x_axis].drop_duplicates(
                ).sort_values().iloc[::2], minor=True)
                plot.get_xaxis().set_minor_formatter('   {x}')
                plot.get_xaxis().set_major_formatter('{x}   ')
                plot.axvline(29, color='k', alpha=.5,
                             linestyle='dashed', zorder=0)
                plot.axvline(56.5, color='k', alpha=.5,
                             linestyle='dashed', zorder=0)
                plot.axvline(86, color='k', alpha=.5,
                             linestyle='dashed', zorder=0)
                plot.grid(axis='y')
            else:
                plot.set_xticks(df[x_axis].drop_duplicates())
                plot.grid()
            plot.legend(title='Lock')
            fig = plot.get_figure()
            fig.savefig(png_dir / (benchmark+suffix + '-' +
                        stats_prefix + 'contention.png'))
            fig.clf()

        # plot = sns.lineplot(
        #     data=df,
        #     estimator=np.median,
        #     x=x_axis, y='iterations_per_process_min',
        #     hue='lock', style='lock', markers=True, dashes=False,
        # )
        # plot = sns.lineplot(
        #     data=df,
        #     estimator=np.median,
        #     x=x_axis, y='iterations_per_process_median',
        #     hue='lock', style='lock', markers=True, dashes=False,
        #     legend=None,
        # )
        # plot = sns.lineplot(
        #     data=df,
        #     estimator=np.median,
        #     x=x_axis, y='iterations_per_process_max',
        #     hue='lock', style='lock', markers=True, dashes=False,
        #     legend=None,
        # )
        # plot.set(
        #     ylim=0,
        #     ylabel='throughput in million locks/s\nwith 95% confidence interval',
        #     xticks=df[x_axis].drop_duplicates(),
        # )
        # fig = plot.get_figure()
        # fig.savefig(png_dir / (benchmark+'-min-median-max.png'))
        # fig.clf()
