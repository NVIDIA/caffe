#ifdef USE_LEVELDB
#ifndef CAFFE_UTIL_DB_LEVELDB_HPP
#define CAFFE_UTIL_DB_LEVELDB_HPP

#include <string>

#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::Iterator* iter)
    : iter_(iter) { SeekToFirst(); }
  ~LevelDBCursor() { delete iter_; }
  void SeekToFirst() override { iter_->SeekToFirst(); }
  void Next() override { iter_->Next(); }
  string key() const override { return iter_->key().ToString(); }
  string value() const override { return iter_->value().ToString(); }
  bool parse(void* datum, DatumTypeInfo datum_type_info) const override {
    if(datum_type_info == DatumTypeInfo_ANNOTATED_DATUM) {
      return static_cast<AnnotatedDatum*>(datum)->ParseFromArray(iter_->value().data(), iter_->value().size());
    } else {
      return static_cast<Datum*>(datum)->ParseFromArray(iter_->value().data(), iter_->value().size());
    }
  }
  const void* data() const override {
    return iter_->value().data();
  }
  size_t size() const override {
    return iter_->value().size();
  }
  bool valid() const override { return iter_->Valid(); }

 private:
  leveldb::Iterator* iter_;
};

class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) { CHECK_NOTNULL(db_); }
  virtual void Put(const string& key, const string& value) {
    batch_.Put(key, value);
  }
  virtual void Commit() {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), &batch_);
    CHECK(status.ok()) << "Failed to write batch to leveldb "
                       << std::endl << status.ToString();
  }

 private:
  leveldb::DB* db_;
  leveldb::WriteBatch batch_;

  DISABLE_COPY_MOVE_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB() : db_(NULL) { }
  virtual ~LevelDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (db_ != NULL) {
      delete db_;
      db_ = NULL;
    }
  }
  virtual LevelDBCursor* NewCursor() {
    return new LevelDBCursor(db_->NewIterator(leveldb::ReadOptions()));
  }
  virtual LevelDBTransaction* NewTransaction() {
    return new LevelDBTransaction(db_);
  }

 private:
  leveldb::DB* db_;
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LEVELDB_HPP
#endif  // USE_LEVELDB
