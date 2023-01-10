====
time
====

Utilities for calculating time operations

 - Can convert delta time from seconds since an epoch to time since a different epoch
 - Can calculate the time in days since epoch from calendar dates

Calling Sequence
----------------

Convert a time from seconds since 1980-01-06T00:00:00 to Modified Julian Days (MJD)

.. code-block:: python

    import pointAdvection.time
    MJD = pointAdvection.time.convert_delta_time(delta_time, epoch1=(1980,1,6,0,0,0),
        epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)

Convert a calendar date into Modified Julian Days

.. code-block:: python

    import pointAdvection.time
    MJD = pointAdvection.time.convert_calendar_dates(YEAR,MONTH,DAY,hour=HOUR,
        minute=MINUTE,second=SECOND,epoch=(1858,11,17,0,0,0))

`Source code`__

.. __: https://github.com/tsutterley/pointAdvection/blob/main/pointAdvection/time.py


General Methods
===============

.. autofunction:: pointAdvection.time.parse_date_string

.. autofunction:: pointAdvection.time.split_date_string

.. autofunction:: pointAdvection.time.datetime_to_list

.. autofunction:: pointAdvection.time.calendar_days

.. autofunction:: pointAdvection.time.convert_datetime

.. autofunction:: pointAdvection.time.convert_delta_time

.. autofunction:: pointAdvection.time.convert_calendar_dates

.. autofunction:: pointAdvection.time.convert_calendar_decimal

.. autofunction:: pointAdvection.time.convert_julian
