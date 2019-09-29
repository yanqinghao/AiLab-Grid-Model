# coding=utf-8
from __future__ import absolute_import, print_function

import os
import sys
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Float, Json
from utils.utils import reCalcHoleGrid, preMain
from utils import get_all_files


@app.input(Folder(key="inputData1", required=True))
@app.input(Json(key="points"))
@app.param(Float(key="heightReal", default=1.0289))
@app.param(Float(key="widthReal", default=1.0294))
@app.output(Folder(key="outputData1", required=True))
def GridPre(context):

    args = context.args

    print(args.inputData1)
    imgs = get_all_files(args.inputData1)

    points = args.points

    for img in imgs:
        imgpath = img.split("/")
        preMain(
            imgpath=img,
            saved_path=os.path.join(args.outputData1, *imgpath[8:-1]),
            model_path=os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])), "utils", "model"
            ),
            height_real=args.heightReal,
            width_real=args.widthReal,
        )
        reCalcHoleGrid(
            png_path=img,
            json_path=os.path.join(args.outputData1, *imgpath[8:-1], "response.json"),
            rectangle=[],
            points=points,
        )

    return args.outputData1


if __name__ == "__main__":
    suanpan.run(GridPre)
