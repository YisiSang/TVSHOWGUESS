# TVShowGuess Dataset #

The dataset consists of 5 popular TV series, including _Friends_, _The Big Bang Theory(TBBT)_, _The Office_, _Frasier_, and _Gilmore Girls_. For each series, the dataset provides character dialogues and backgrounds descriptions. The characters' dialogues start with the characters' names. One or more rounds of dialogue between characters form a scene. Scenes are separated by short backgrounds that begin with markers such as location (e.g. **Howard's car**, **Kingman Police Station**), special words (e.g., **Scene**, **Cut**), or symbols (e.g. **[ ]**). To extract information related to our task (i.e., independent scenes) in a structured form, we created a rule-based parser which splits the content of an episode into multiple independent scenes using scene separation markers.

In this repository, we only release the processed data.


## Data Download ##

| Name                  | <div style="width:150px">Description</div>                            |  Link                |  
| :-------------------- | :-------------------------------------------------------------------- | :------------------: |  
| Split Scenes	        | Idenfify the scenes in the raw data and generate a list of scenes     | [OneDrive](https://1drv.ms/u/s!ArPzysVAJSvtqKJZipldfI7k9SZ0cA?e=5fXvmT) |
| Merged by Character   | The utterances/narratives are re-organized by speakers and background | [OneDrive](https://1drv.ms/u/s!ArPzysVAJSvtqKJXkM02lpXowgA9-g?e=nqTQyL) |  

Our provided scripts use the __split_scenes__ files for tokenization and model training. We also merge the data by characters upon the split scenes for your best convenience. 

## Data Format ##

### Split Scenes ###
Each json file is an episode.
```json
{
    "Episode Number": "01x01",
    "Episode Title": "The Good Son",
    "Recap": [
        "Dr. Frasier Crane, formerly..."
    ],
    "Show Title": "Frasier",
    "Transcript": [
        "Act One.",
        "THE JOB",
        "Scene One ...",
        "Frasier: ...",
        ...
    ],
    "Transcript Author": "bunniefuu",
    "scenes": [
        {
            "id": 1,
            "title": "Act One.",
            "lines": [
                [
                    "background",
                    "THE JOB"
                ]
            ],
            "participants": {}
        },
        {
            "id": 2,
            "title": "Scene One ...",
            "lines": [
                [
                    "frasier",
                    "Frasier: ..."
                ],
                ...
            ],
            "participants": {
                "frasier": 8,   // # of utterances in the scene
                "roz": 6,       // # of utterances in the scene
                "russell": 1    // # of utterances in the scene
            }
        },
        ...
    ]
}
```

### Merged by Character ###
Each line in the show-specific **.json** files represents a section, containing a list of scenes and backgrounds.
```json
{
    "id": 2,
    "title": "Scene One ...",
    "lines": [
        [
            "frasier",
            "Frasier: ..."
        ],
        ...
        [
            "background",
            "Roz speaks in a soothing radio voice."
        ],
        ...
    ],
    "participants": {
        "frasier": 8,
        "roz": 6,
        "russell": 1
    },
    "episode_id": "01x01"
}
```
Note that there are some empty sections that include no scenes or backgrounds in the show-specific **.json** files. They are simply skipped.



