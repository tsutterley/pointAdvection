import pointCollection as pc

class velocity(pc.grid.data):
    def __init__(self):
        self.t=None
        super(velocity, self).__init__()
        
    def from_file(self, filename, format='NSIDC-0725', **kwargs):
        
        if format in ['NSIDC-0725','NSIDC-0727', 'NSIDC-0731', 'NSIDC-0766']:
            field_dict={'U':vx,'V':vy,'eU':'ex','eV':'ey'}
            self=self.from_h5(filename, field_mapping=field_dict, **kwargs)
        elif format=='NSIDC-0720' or format=='NSIDC-0484':
            field_dict={'U':'VX','V':'VY','eU':'ERRX','eV':'ERRY'}
            self=self.from_nc(filename, field_mapping=field_dict, **kwargs)
        else:
            print(f"format {format} for file {filename} unknown, skipping")
        return self

    def interp_to(self, x0, y0, t0=None):

        temp={'x':x0,
            'y':y0,
            'time':t0}
        for field in self.fields:
            temp[field]=self.interp(x0.ravel(), y0.ravel(), t0, field=field, gridded=True)
        return velocity().from_dict(temp)