#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <stdint.h>
#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor)
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
  }
  void SeekToFirst() override { Seek(MDB_FIRST); }
  void Next() override { Seek(MDB_NEXT); }
  string key() const override {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  string value() const override {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  bool parse(void* datum, DatumTypeInfo datum_type_info) const override {
    if(datum_type_info == DatumTypeInfo_ANNOTATED_DATUM) {
      return static_cast<AnnotatedDatum*>(datum)->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    } else {
      return static_cast<Datum*>(datum)->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    }
  }
  const void* data() const override {
    return mdb_value_.mv_data;
  }
  size_t size() const override {
    return mdb_value_.mv_size;
  }

  bool valid() const override { return valid_; }

 private:
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
};

class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_env* mdb_env)
    : mdb_env_(mdb_env) { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit();

 private:
  MDB_env* mdb_env_;
  vector<string> keys, values;

  void DoubleMapSize();

  DISABLE_COPY_MOVE_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL), mdb_dbi_() { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual LMDBCursor* NewCursor();
  virtual LMDBTransaction* NewTransaction();

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
