#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string_view>
#include <vector>

namespace lczero {

template <class T>
T kind_of_bit_cast(std::uint64_t from) {
  T to;
  std::memcpy(&to, &from, sizeof(to));
  return to;
}

template <class T>
class ProtoIterator {
 public:
  class Iter {
   public:
    Iter(std::function<T(size_t)> func, size_t idx)
        : accessor_(func), idx_(idx) {}
    void operator++() { ++idx_; }
    T operator*() { return accessor_(idx_); }
    bool operator!=(const Iter& other) { return idx_ != other.idx_; }

   private:
    const std::function<T(size_t)> accessor_;
    size_t idx_;
  };

  ProtoIterator(size_t count, std::function<T(size_t)> accessor)
      : count_(count), accessor_(accessor) {}

  Iter begin() const { return Iter(accessor_, 0); }
  Iter end() const { return Iter(accessor_, count_); }

 private:
  const size_t count_;
  const std::function<T(size_t)> accessor_;
};

class ProtoMessage {
 public:
  void ParseFromString(const std::string&);

  class Builder {
   public:
    Builder();
    std::string AsString() const;

   protected:
    void operator=(const ProtoMessage& msg);
    Builder(const ProtoMessage&);
    void WireFieldClear(int wire_field_id);
    void WireFieldSetVarint(int wire_field_id, std::uint64_t varint);
    void WireFieldSetMessage(int wire_field_id, const ProtoMessage& val);

   private:
    using Bits = std::vector<std::string>;
    using Fields = std::map<int, Bits>;
    Fields fields_;
  };

 protected:
  ProtoMessage() = default;
  ProtoMessage(const ProtoMessage& other);
  ProtoMessage(ProtoMessage&& other);
  void operator=(const ProtoMessage& other);
  void operator=(ProtoMessage&& other);
  ProtoMessage(const Builder&);
  void operator=(const Builder&);
  static constexpr size_t kLast = std::numeric_limits<size_t>::max();

  ProtoMessage(std::string_view serialized_proto);
  size_t WireFieldCount(int wire_field_id) const;
  std::uint64_t GetVarintVal(int wire_field_id, size_t index) const;
  float GetFloatVal(int wire_field_id, size_t index) const;
  double GetDoubleVal(int wire_field_id, size_t index) const;
  std::uint32_t GetFixed32Val(int wire_field_id, size_t index) const;
  std::uint64_t GetFixed64Val(int wire_field_id, size_t index) const;
  std::string_view GetBytesVal(int wire_field_id, size_t index) const;

 private:
  void RebuildOffsets();
  const char* GetFieldPtr(int wire_field_id, size_t index) const;

  struct FieldPos {
    size_t offset;
    size_t size;
  };
  using Offsets = std::vector<FieldPos>;
  using FieldOffsets = std::map<int, Offsets>;

  FieldOffsets offsets_;
  std::string buf_owned_;
  std::string_view buf_unowned_;
};

}  // namespace lczero