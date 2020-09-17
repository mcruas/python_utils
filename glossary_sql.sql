-- Converter int em data

DECLARE @DT_INI_WINDOW INT = 20191001
print CONVERT(date,convert(char(8),@DT_INI_WINDOW ))

