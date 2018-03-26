#!/bin/bash

scp -r -i ~/Documents/CourseWork/Cloud\ Computing/15319demo.pem ~/Documents/MyResume/OperationsResearch/Heavywater/datasets ubuntu@ec2-34-203-191-29.compute-1.amazonaws.com:~/flaskapp/
scp -r -i ~/Documents/CourseWork/Cloud\ Computing/15319demo.pem ~/Documents/MyResume/OperationsResearch/Heavywater/templates ubuntu@ec2-34-203-191-29.compute-1.amazonaws.com:~/flaskapp/
