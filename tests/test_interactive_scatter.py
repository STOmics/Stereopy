from stereo.io.reader import read_stereo
# import pandas as pd
# import panel as pn
# from stereo.plots.interactive_scatter import InteractiveScatter

gem = 'D:\\projects\\data\\st_demo\\DP8400013846TR_F5.gem'
data = read_stereo(gem, 'bins', 20)

# iscatter = InteractiveScatter(data)
# scatter = iscatter.interact_scatter()
# scatter.show(threaded=True)

inter_s = data.interact_scatter(inline=False)
# inter_s.show(threaded=True)
# pn.template.FastListTemplate(site="Panel", title="interactive scatter", main=[scatter]).servable()
