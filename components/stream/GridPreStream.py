# coding=utf-8
from __future__ import absolute_import, print_function

import os
import sys
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Json
from utils.utils import reCalcHoleGrid, preMain


@app.input(Json(key="inputData1"))
# @app.input(Image(key="inputData1", required=True))
# @app.input(Json(key="points"))
# @app.param(Float(key="heightReal", default=1.0289))
# @app.param(Float(key="widthReal", default=1.0294))
@app.output(Json(key="outputData1"))
def GridPre(context):

    args = context.args
    inputData = args.inputData1
    imagePath = inputData["image"]
    points = inputData["points"]
    heightReal = inputData["heightReal"]
    widthReal = inputData["widthReal"]

    inputFolder = os.path.split(imagePath)[0]
    img = os.path.split(imagePath)[1]
    preMain(
        imgpath=imagePath,
        saved_path=os.path.join(inputFolder, "result_" + img),
        model_path=os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), "utils", "model"
        ),
        height_real=heightReal,
        width_real=widthReal,
    )
    reCalcHoleGrid(
        png_path=imagePath,
        json_path=os.path.join(inputFolder, "result_" + img, "response.json"),
        rectangle=[],
        points=points,
    )

    outputData = {"files": os.path.join(inputFolder, "result_" + img)}

    return outputData


if __name__ == "__main__":
    suanpan.run(GridPre)
