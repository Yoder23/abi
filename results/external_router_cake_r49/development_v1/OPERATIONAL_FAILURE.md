# R49 v1 operational failure

The router fit completed and passed 1,400/1,400 search plus 1,400/1,400
validation route decisions.  The run then failed closed before model loading or
generation because the implementation required exact equality between the
selected 1,400 catalog IDs and a 1,600-ID source-bundle superset.  There were
zero missing selected IDs and 200 surplus historical IDs.  No integrated row
was generated, so this is not a scientific model result.

- router JSON SHA-256: `f9e43bf4240ee663538e71685261f22237954b6717d36f7c7fd4bf987e154670`
- router package SHA-256: `20ae6562432cc59ece5b9cb552caaa16da4824323e62460c6c33feb0579df863`
