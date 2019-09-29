# coding=utf-8
from __future__ import absolute_import, print_function

import os
import sys
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Json
from utils.utils import reCalcHoleGrid, preMain
from suanpan.storage import storage
from suanpan.log import logger


@app.input(Json(key="inputData1"))
@app.output(Json(key="outputData1"))
def GridPreStream(context):

    args = context.args
    inputData = args.inputData1
    func = inputData["func"]
    imageUrl = inputData["image"]
    if "points" in inputData.keys():
        points = inputData["points"]
    if "heightReal" in inputData.keys():
        heightReal = inputData["heightReal"]
    if "widthReal" in inputData.keys():
        widthReal = inputData["widthReal"]
    if "rectangle" in inputData.keys():
        rectangle = inputData["rectangle"]

    imagePath = os.path.join("/spnext", imageUrl)

    storage.download(imageUrl, imagePath)

    inputFolder = os.path.split(imagePath)[0]
    img = os.path.split(imagePath)[1]
    if func == "preMain":
        preMain(
            imgpath=imagePath,
            saved_path=os.path.join(inputFolder, "result_" + img),
            model_path=os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])), "utils", "model"
            ),
            height_real=heightReal,
            width_real=widthReal,
        )
    elif func == "reCalcHoleGrid":
        reCalcHoleGrid(
            png_path=imagePath,
            json_path=os.path.join(inputFolder, "result_" + img, "response.json"),
            rectangle=rectangle,
            points=points,
        )
    else:
        logger.error("Undefined Function")
        raise Exception("Undefined Function")

    storage.upload(
        os.path.join(imageUrl, "result"), os.path.join(inputFolder, "result_" + img)
    )

    outputData = {"files": os.path.join(imageUrl, "result_" + img)}

    return outputData


if __name__ == "__main__":
    suanpan.run(app)
