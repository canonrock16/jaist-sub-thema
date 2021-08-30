# jaist-sub-thema

副テーマ研究用リポジトリ

# commands

- コンテナ生成

```bash
docker-compose up
# ビルド強制
docker-compse up --build
```

- コンテナに入る

```
docker exec -it sub-thema-main /bin/bash
```

- jupyterlab 立ち上げ

```bash
# ip=0.0.0.0が必要。127.0.0.1は仮想ネットワークインターフェースなので、立ち上げているhost外からはアクセスできない。0.0.0.0は全てのネットワークインターフェースを指し、host外からでもアクセス可能。
jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

- MPI 並列処理の実行

```bash
# 自宅PCは最大8コア。それ以上を指定するとエラー
mpiexec -n 8 --allow-run-as-root python paralleled_collaborative_filtering.py
```
