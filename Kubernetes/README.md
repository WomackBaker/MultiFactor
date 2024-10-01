# To run kubernetes server

- Change yaml file to run docker files
- Start kubernetes 
- Run kubectl apply -f android.yaml -f auth.yaml -f gen.yaml -f log.yaml

# To stop kubernetes server
- Run kubectl delete -f android.yaml -f auth.yaml -f gen.yaml -f log.yaml