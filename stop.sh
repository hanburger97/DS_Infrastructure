#!/bin/bash

ps -ef | grep jupyter | grep -v grep | awk '{print $2}' | xargs kill

