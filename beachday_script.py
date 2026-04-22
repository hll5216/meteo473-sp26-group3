#Imports
from herbie import Herbie, FastHerbie
import pandas as pd, numpy as np
import xarray as xr
import dask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os

#Output folder
DATA_FILE = "beachday_new.nc"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Define index
def beachday_index(ds,ts):
    ds_time=ds.isel(valid_time=ts)

    temp_f = (ds_time['t2m'].values - 273.15)*(9/5) + 32
    temp_contribution = np.zeros_like(temp_f)
    mask = (temp_f >= 60) & (temp_f < 75)
    temp_contribution[mask] = (2/3)*(temp_f[mask] - 60)
    mask = (temp_f >= 75) & (temp_f <= 85)
    temp_contribution[mask] = 10.0
    mask = (temp_f > 85) & (temp_f < 100)
    temp_contribution[mask] = 10*(0.25)**((temp_f[mask]-85)/15) 
    mask = temp_f >= 100
    temp_contribution[mask] = 2.5

    dewpoint_f = (ds_time['d2m'].values - 273.15)*(9/5) + 32
    dewpoint_f.max() 
    dewpoint_contribution = np.zeros_like(dewpoint_f)
    mask = (dewpoint_f<45)
    dewpoint_contribution[mask]=2.5
    mask = (dewpoint_f >= 45) & (dewpoint_f < 55)
    dewpoint_contribution[mask] = 0.004488*(1.1487)**(dewpoint_f[mask])
    mask = (dewpoint_f >= 55) & (dewpoint_f <= 75)
    dewpoint_contribution[mask] = 10.0
    mask = (dewpoint_f > 75) & (dewpoint_f < 85)
    dewpoint_contribution[mask] = 11*np.exp(-0.2398*(dewpoint_f[mask]-75))-1
    mask = dewpoint_f >= 85
    dewpoint_contribution[mask] = 0

    totalcloud = (ds_time['tcc'].values)
    totalcloud.max() 
    cloud_contribution = np.zeros_like(totalcloud)
    mask = (totalcloud >= 0) & (totalcloud <= 35)
    cloud_contribution[mask] = 10
    mask = (totalcloud > 35) & (totalcloud < 80)
    cloud_contribution[mask] = ((-2/9)*(totalcloud[mask]))+(160/9)
    mask = (totalcloud >= 80) & (totalcloud <= 100)
    cloud_contribution[mask] = 0

    u=(ds_time['u10'].values)
    v=(ds_time['v10'].values)
    wind= np.sqrt(u**2+v**2)
    wind_mph= wind*2.23693629 
    wind_mph.max()
    wind_contribution = np.zeros_like(wind_mph)
    mask = (wind_mph >= 0) & (wind_mph < 10)
    wind_contribution[mask] = 10
    mask = (wind_mph > 10) & (wind_mph < 20)
    wind_contribution[mask] =121*np.exp(-0.24*(wind_mph[mask]))-1
    mask = (wind_mph >= 20) 
    wind_contribution[mask] = 0
    mask = wind_contribution < 0
    wind_contribution[mask] = 0

    prate_mm = ds_time['prate'].values
    prate_mm_hr = prate_mm*3600
    prate_in_hr = prate_mm_hr*0.0393701
    precip_contribution = np.zeros_like(prate_in_hr)
    mask = (prate_in_hr >= 0) & (prate_in_hr <= 0.2)
    precip_contribution[mask] = (-50*(prate_in_hr[mask]))+10

    BD_index = (temp_contribution*0.3+dewpoint_contribution*.005+cloud_contribution*.10+wind_contribution*.15+precip_contribution*.40)
    return BD_index

#Define single plot function
def plot_index(ds, bd, timestep, init_time, valid_time, save=True):
    #Create plot and colorbar
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_title("Beach Day Index", loc='left', fontsize=12)
    ax.set_title(f'Init: {init_time}\nValid: {valid_time}', loc='right', fontsize=12)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_extent([-100, -60, 25, 50], crs=ccrs.PlateCarree())
    levels = np.arange(0, 10.25, 0.25)
    final_index = ax.contourf(ds['longitude'].values, ds['latitude'].values, bd, levels=levels, cmap='RdYlGn', transform=ccrs.PlateCarree())
    index_cb = fig.colorbar(final_index, ax=ax, orientation='horizontal', pad=0.05)
    plt.tight_layout()

    #Save figure as .png file
    file_name = os.path.join(OUTPUT_DIR, f"beachday_{timestep:03d}.png")
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {file_name}")

#Automation
print("Loading dataset...")
ds = xr.open_dataset(DATA_FILE)
init_time = pd.to_datetime(ds.time.values).strftime('%HZ %b %d %Y')
for t in range(ds.sizes['valid_time']):
    bd = beachday_index(ds, t)
    valid_time = pd.to_datetime(ds.valid_time.values[t]).strftime('%HZ %b %d %Y')
    plot_index(ds, bd, t, init_time, valid_time)
print("All plots saved.")