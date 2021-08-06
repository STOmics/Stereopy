import pandas as pd
import panel as pn
from stereo.io.reader import read_stereo
from stereo.plots.interactive_scatter import InteractiveScatter

gem = 'D:\\projects\\data\\st_demo\\DP8400013846TR_F5.gem'
data = read_stereo(gem, 'bins', 50)

iscatter = InteractiveScatter(data)
scatter = iscatter.interact_scatter()

pn.template.FastListTemplate(site="Panel", title="interactive scatter", main=[scatter]).servable()

