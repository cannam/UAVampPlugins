/****************
Copyright (c) 2005-2012 Antonio Pertusa Ibáñez <pertusa@dlsi.ua.es>
Copyright (c) 2012 Universidad de Alicante

This onset detection system is free software: you can redistribute it and/or
 modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your
 option) any later version.

This onset detection system is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code.  If not, see <http://www.gnu.org/licenses/>.

 Comments are welcomed
 
**************/

#include "myfft.h"
#include "params.h"

#include <vamp-sdk/FFT.h>

#include <iostream>
#include <mutex>
#include <map>
#include <memory>
#include <vector>

using namespace std;
using Vamp::FFTReal;

static const Float w_rate = (TWO_PI / 22050.0);


/*--------------------------------------------

 Conversion functions

---------------------------------------------*/

Float mus_linear_to_db(Float x) {
	if (x > 0.0) return(20.0 * log10(x)); 
	return(-200.0);
}

Float mus_radians_to_hz(Float rads) {return(rads / w_rate);}
Float mus_hz_to_radians(Float hz) {return(hz * w_rate);}
Float mus_degrees_to_radians(Float degree) {return(degree * TWO_PI / 360.0);}
Float mus_radians_to_degrees(Float rads) {return(rads * 360.0 / TWO_PI);}
Float mus_db_to_linear(Float x) {return(pow(10.0, x / 20.0));}


/*--------------------------------------------

 IsNaN returns 1 if the input number is a NaN.

-------------------------------------------*/

bool isNaN(double val) {
	return (isnan(val) || isinf(val));
}

/*---------------------------------------------

 Create Hanning window

----------------------------------------------*/

void Hanning(Float *window, int size) {
	for (int i=1; i<=size; i++)
		window[i-1]=0.5*(1-cos(2*M_PI*((double)i/(double)(size+1))));
}

/*--------------------------------------------

 Apply FFT

---------------------------------------------*/

void mus_fft(Float *rl, int n)
{
    static map<int, shared_ptr<FFTReal>> ffts;
    static vector<double> inbuf;
    static vector<double> outbuf;
    static mutex mut;
    lock_guard<mutex> guard(mut);

    if (ffts.find(n) == ffts.end()) {
        ffts[n] = make_shared<FFTReal>(n);
        if (n > int(inbuf.size())) {
            inbuf.resize(n);
            outbuf.resize(n+2);
        }
    }

    for (int i = 0; i < n; ++i) {
        inbuf[i] = rl[i];
    }

    ffts.at(n)->forward(inbuf.data(), outbuf.data());

    // Output is expected to match FFTW's R2HC format
    
    for (int i = 0; i <= n/2; ++i) {
        rl[i] = outbuf[i*2];
    }
    for (int i = 1; i < n/2; ++i) {
        rl[n-i] = outbuf[i*2+1];
    }
}

/*---------------------------------------------
 
 Spectrum calculation, using mus_fft

---------------------------------------------*/

void fourier_spectrum(Float *sf, Float *fft_data, int fft_size, int data_len, Float *window, int win_len)
{
  int i;

  if (window)
  {
      for (i = 0; i < win_len; i++)
	fft_data[i] = window[i] * sf[i]; 

      for (i=win_len; i<data_len; i++)
	fft_data[i] = sf[i];
  }
  else
  {
      for (i = 0; i < data_len; i++)
	fft_data[i] = sf[i];
  }

  if (data_len < fft_size) 
      memset((void *)(fft_data + data_len), 0, (fft_size - data_len) * sizeof(Float));

  int j;
  mus_fft(fft_data, fft_size);

  fft_data[0] = fabs(fft_data[0]);
  fft_data[fft_size / 2] = fabs(fft_data[fft_size / 2]);

  for (i = 1, j = fft_size - 1; i < fft_size / 2; i++, j--) {
     fft_data[i] = hypot(fft_data[i], fft_data[j]);
  }

  // Correct NaN values

  for (int i=0; i<fft_size/2; i++)
	if (isNaN(fft_data[i]))
		fft_data[i]=0.0;
}

