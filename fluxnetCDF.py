import os
import re
from datetime import datetime
import glob
import numpy as np
import pandas as pd

""" 
This script converts .CSV files from FLUXNET2015 Tier 2 to netCDF.
Written by Marco Hannemann (marco.hannemann@ufz.de)

0. Prequisites
    
Unzip the .ZIP files from the FLUXNET2015 database to your desired folder and make sure each station has its own
directory containing the CSVs; e.g.:

    E:/Data/FLUXNET2015/FLX_AT-Neu_FLUXNET2015_FULLSET_2002-2012_1-4
    E:/Data/FLUXNET2015/FLX_CH-Cha_FLUXNET2015_FULLSET_2005-2014_2-4

Both files for legend and site info should be present in the doc/ folder containing FLUXNET2015 metadata.
Additionally, docs/variable_groups.csv lists the variable groups required for filtering the variable output.
    
1. Select the variables you want to write to netCDF by setting the flags in output_variables.csv to 0 or 1.
2. Set the attributes in the main function at the end of this script (see main function header for detailed
    explanation of the attributes. 
3. Run program and profit
"""


class Error(Exception):
    """Base class for other exceptions"""
    pass


class SiteCodeError(Error):
    """Invalid site code"""

    def __init__(self, site, message=f"File with selected site code does not exist or multiple directories exist"):
        self.sitecode = site
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.sitecode}: {self.message}'


class SetTypeError(Error):
    """Invalid set type"""

    def __init__(self, settype, message=f"Set type is not valid or file does exist. Choose from ['FULLSET'|'SUBSET'] "):
        self.settype = settype
        self.message = message
        super().__init__(self.message)


class TemporalResolutionError(Error):
    """Invalid temporal aggregation selected"""

    def __init__(self, temporal_agg,
                 message=f"Selected temporal aggregation is invalid. Choose from ['HH'|'DD'|'WW'|'YY']"):
        self.temporal_agg = temporal_agg
        self.message = message
        super().__init__(self.message)


def check_date_format(dframe, time_col_name):
    """Checks format of timestamp and returns strftime format"""
    date_digit_length = len(str(dframe[time_col_name][0]))
    if date_digit_length == 4:
        date_format = '%Y'
    elif date_digit_length == 8:
        date_format = '%Y%m%d'
    elif date_digit_length == 12:
        date_format = '%Y%m%d%H%M'
    return date_format


def find_flux_file(path, site, temporal_agg, settype):
    """Finds specific FLUXNET2015 station CSV data from folder containing several station CSV files."""
    if settype not in ['FULLSET', 'SUBSET']:
        raise SetTypeError(settype)
    flux_dir = [dir for dir in os.listdir(path) if site in dir]
    if len(flux_dir) != 1:
        raise SiteCodeError

    files = [os.path.basename(file) for file in glob.glob(path + flux_dir[0] + '/*.csv')]
    files = [file for file in files if file.split('_')[1] == site and file.split('_')[3] == settype]
    fname = [file for file in files if file.split('_')[4] == temporal_agg]
    return flux_dir[0] + '/' + fname[0]


def filter_output(df):
    """Filters output variables to be written based on flags in output_variables.csv."""
    variable_groups = pd.read_csv('doc/variable_groups.csv', sep=';')
    output_variables = pd.read_csv('output_variables.csv', sep='\t')

    # filter all variables from groups flagged with 1
    output_groups = output_variables.loc[output_variables['FLAG'] == 1]['GROUP'].values
    output_variables = variable_groups[variable_groups['Group'].isin(output_groups)]['Variable']

    # Some variables are numbered like TA_F_MDS_1, but listed as TA_F_MDS_#. REGEX are used to filter them out
    numbered_variables = []
    for numbered_variable in list(output_variables[output_variables.str.contains('#')]):
        numbered_variables.append(list(filter(re.compile(numbered_variable.replace('#', r'\d')).match, df.columns)))
    numbered_variables = [item for subitem in numbered_variables for item in subitem]
    output_variables = [*output_variables, *numbered_variables]

    # filter out q
    if 'QUALITY FLAGS' not in output_groups:
        output_variables = [v for v in output_variables if 'QC' not in v]

    df = df.drop(columns=df.columns.difference(list(output_variables)), axis=1, errors='ignore')
    return df


def flux2netcdf(path, site, temporal_agg, settype):

    if temporal_agg not in ['HH', 'DD', 'WW', 'YY']:
        raise TemporalResolutionError(temporal_agg)

    fname = find_flux_file(path, site, temporal_agg, settype)

    try:
        df = pd.read_csv(os.path.join(path, fname))
        metadata = site_info.loc[site]
    except (ValueError, IndexError):
        raise SiteCodeError(site)

    # Set time variable name depending on temporal aggregation

    if temporal_agg == 'HH':
        time_col_name = 'TIMESTAMP_START'
        df.drop(columns='TIMESTAMP_END', inplace=True)
    else:
        time_col_name = 'TIMESTAMP'

    # Set DateTimeIndex
    date_format = check_date_format(df, time_col_name)
    df[time_col_name] = pd.to_datetime(df[time_col_name], format=date_format)
    df.set_index(time_col_name, inplace=True)
    df.index = df.index.rename('time')

    # Filter out variable groups to be written
    df = filter_output(df)

    # Convert DataFrame to xarray
    ds = df.to_xarray()

    # Set variable attributes from metadata
    for variable in list(ds.data_vars):
        if variable in legend['Variable'].values:
            v = variable
        # Variables from night time partitioning method need special handling because of dynamic threshold XX
        elif variable.split('_')[-2] == 'CUT' and variable.split('_')[-1].isdigit():
            v = variable[:-2] + 'XX'
        # Units for different temporal aggregations must be extracted from following 3 lines
        if np.isnan(legend.iloc[legend[legend['Variable'] == v].index]['Units'].values[0]):
            unit_block = legend.iloc[(legend[legend['Variable'] == v].index[0] + 1):(
                    legend[legend['Variable'] == v].index[0] + 4)]
            try:
                unit = unit_block[unit_block['Variable'] == temporal_agg].Units.values[0]
            except IndexError:
                unit = ''
        # Simple case, unit present independent from temporal aggregation
        else:
            unit = legend.iloc[legend[legend['Variable'] == v].index]['Units'].values[0]

        ds[variable].attrs = {'long_name': legend[legend['Variable'] == v].Description.values[0],
                              'units': unit,
                              '_FillValue': -9999.0,
                              }

    # add lat and lon coordinates to data set
    ds['lon'] = metadata.loc['lon']
    ds['lat'] = metadata.loc['lat']
    ds['lon'].attrs = {'standard_name': 'longitude', 'long_name': 'longitude coordinate',
                       'units': 'degrees_east', '_FillValue': False}
    ds['lat'].attrs = {'standard_name': 'latitude', 'long_name': 'latitude coordinate',
                       'units': 'degrees_north', '_FillValue': False}

    # ds = ds.assign_coords({'lat': ds.lat, 'lon': ds.lon}).expand_dims(['lat', 'lon'], axis=(1,2))

    ds.time.attrs = {'long_name': 'time', '_FillValue': False}

    # Set global attributes
    ds.attrs = {f'title': f'FLUXNET 2015 Tier 2 {site} {temporal_agg}',
                'content': f'Flux station {site}',
                'country': metadata.loc['country'],
                'elevation': metadata.loc['elev'],
                'longitude': metadata.loc['lon'],
                'latitude': metadata.loc['lat'],
                'plant_functional_type': metadata.loc['pft'],
                'institution': 'https://fluxnet.org',
                'source': 'FLUXNET',
                'history': f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
                           f'Python 3.9 fluxnetCDF.py main({path}, {site}, {temporal_agg}, {settype}',
                'contact': 'marco.hannemann@ufz.de',
                'comment': 'Data converted from raw .CSV data to netCDF without any '
                           'further processing steps.'}

    # comp = dict(zlib=True, complevel=5)
    # encoding = {var: comp for var in ds.data_vars}
    # encoding['time'] = {'dtype': 'double'}
    try:
        os.mkdir('netcdf')
    except FileExistsError:
        pass

    ds.to_netcdf(f'netcdf/FLX_{site}_{temporal_agg}.nc', encoding={'time': {'dtype': 'double'}},
                 unlimited_dims='time')


def main(path, site, temporal_agg, settype):
    """
    :param path: str
        path to directory containing folders of FLUXNET2015 Tier 2 data
    :param site: str
        Site code in format 'AA-Flx' OR 'all' to convert all data sets found in path
    :param temporal_agg: str
        Temporal aggregation with HH: Hourly, DD: Daily, YY: Yearly
    :param settype: str
        Data set type, FULLSET or SUBSET
    """

    if site == 'all':
        all_sites = [x[4:10] for x in os.listdir(path) if x[0:3] == 'FLX']
        for site in all_sites:
            print(site)
            if site == 'AA-Flx':
                continue
            flux2netcdf(path, site, temporal_agg, settype)
    else:
        flux2netcdf(path, site, temporal_agg, settype)


legend = pd.read_csv('doc/variable_codes_FULLSET_20200504.csv')
site_info = pd.read_csv('doc/site_info.csv', index_col='site')

main(path='E:/Data/FLUXNET2015/',
     site='all',
     temporal_agg='DD',
     settype='FULLSET')
