/*
  Originally from jzip, (http://jzip.sourceforge.net/) distributed under
  the modified BSD license.
  Copyright (c) 2000  John D. Holder.

  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "zmachine/zmachine.h"

namespace lczero {
namespace {

const OptionId kZFileId{"z-file", "", "Zcode file to load.", 'z'};
const OptionId kColorId{"color", "", "Render text in color."};
const OptionId kHistBufId{"hist-buf-size", "", "Size of the history buffer."};
const OptionId kUnicodeId{"unicode", "", "How to display and handle unicode text."};

void configure( zbyte_t min_version, zbyte_t max_version )
{
   zbyte_t header[PAGE_SIZE]/*, second*/;

   read_page( 0, header );
   datap = header;

   h_type = get_byte( H_TYPE );

#if 0
   if ( h_type == 'M' || h_type == 'Z' )
   {
      /* possibly a DOS executable file, look more closely. (mol951115) */
      second = get_byte( H_TYPE + 1 );
      if ( ( h_type == 'M' && second == 'Z' ) || ( h_type == 'Z' && second == 'M' ) )
         if ( analyze_exefile(  ) )
         {
            /* Bingo!  File is a standalone game file.  */
            /* analyze_exefile has updated story_offset, so now we can */
            /* read the _real_ first page, and continue as if nothing  */
            /* has happened. */
            read_page( 0, header );
            h_type = get_byte( H_TYPE );
         }
   }
#endif
   GLOBALVER = h_type;

   if ( h_type < min_version || h_type > max_version ||
        ( get_byte( H_CONFIG ) & CONFIG_BYTE_SWAPPED ) )
      fatal( "Wrong game or version" );
   /*
    * if (h_type == V6 || h_type == V7)
    * fatal ("Unsupported zcode version.");
    */

   if ( h_type < V4 )
   {
      story_scaler = 2;
      story_shift = 1;
      property_mask = P3_MAX_PROPERTIES - 1;
      property_size_mask = 0xe0;
   }
   else if ( h_type < V8 )
   {
      story_scaler = 4;
      story_shift = 2;
      property_mask = P4_MAX_PROPERTIES - 1;
      property_size_mask = 0x3f;
   }
   else
   {
      story_scaler = 8;
      story_shift = 3;
      property_mask = P4_MAX_PROPERTIES - 1;
      property_size_mask = 0x3f;
   }

   h_config = get_byte( H_CONFIG );
   h_version = get_word( H_VERSION );
   h_data_size = get_word( H_DATA_SIZE );
   h_start_pc = get_word( H_START_PC );
   h_words_offset = get_word( H_WORDS_OFFSET );
   h_objects_offset = get_word( H_OBJECTS_OFFSET );
   h_globals_offset = get_word( H_GLOBALS_OFFSET );
   h_restart_size = get_word( H_RESTART_SIZE );
   h_flags = get_word( H_FLAGS );
   h_synonyms_offset = get_word( H_SYNONYMS_OFFSET );
   h_file_size = get_word( H_FILE_SIZE );
   if ( h_file_size == 0 )
      h_file_size = get_story_size(  );
   h_checksum = get_word( H_CHECKSUM );
   h_alternate_alphabet_offset = get_word( H_ALTERNATE_ALPHABET_OFFSET );

   datap = NULL;
}

}  // namespace

void ZMachine::Run() {

  OptionsParser options;

  options.Add<StringOption>(kZFileId) = "zugzwang.z5";
  options.Add<IntOption>(kHistBufId, 0, 16384) = 1024;
  options.Add<BoolOption>(kColorId) = false;
  std::vector<std::string> unicode_opts = {"none", "zscii", "full"};
  options.Add<ChoiceOption>(kUnicodeId, unicode_opts) = "full";

  if (!options.ProcessAllFlags()) return;

  auto option_dict = options.GetOptionsDict();

  monochrome = option_dict.Get<bool>(kColorId.GetId()) ? 0 : 1;

  if (option_dict.Get<std::string>(kUnicodeId.GetId()) == "none") {
    unicode = 0;
  } else if (option_dict.Get<std::string>(kUnicodeId.GetId()) == "zscii") {
    unicode = 1;
  } else {
    unicode = 2;
  }

  hist_buf_size = option_dict.Get<int>(kHistBufId.GetId());

  open_story(option_dict.Get<std::string>(kZFileId.GetId()).c_str());

  configure( V1, V8 );

  initialize_screen(  );

  load_cache(  );

  z_restart(  );

  ( void ) interpret(  );

  close_story(  );

  close_script(  );

  unload_cache(  );

  reset_screen(  );

}


}  // namespace lczero
