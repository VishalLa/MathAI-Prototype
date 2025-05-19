from roboflow import Roboflow
rf = Roboflow(api_key="7jvjOy6PV6DBkLdn91FY")
project = rf.workspace().project("test_project-wv1wl")
version = project.version(1)
dataset = version.upload("C:\\Users\\visha\\OneDrive\\Desktop\\new dataset", num_threads=8)
