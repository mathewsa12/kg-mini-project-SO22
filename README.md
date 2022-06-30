# Mini-Project - Foundations of Knowledge Graphs

## Classification of the remaining individuals from carcinogenesis

#### Fact Validators

Team Members (IMT Username):
Aabha Baboo Mathews (mathewsa)
Palaniappan Muthuraman (palani)

To clone the git repo:
`git clone https://github.com/mathewsa12/kg-mini-project-SO22.git`

To build and run the docker:
`docker build -t <container_name> .`
`docker run <container_name> `

To copy the file from the image to the current directory:
1. `docker ps -a`
2. `docker cp <container_id>:classification_result.ttl .`